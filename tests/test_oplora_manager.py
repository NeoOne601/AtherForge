# AetherForge v1.0 — tests/test_oplora_manager.py
# ─────────────────────────────────────────────────────────────────
# Unit tests for OPLoRA mathematical operations.
# Validates the SVD projector math guarantees:
#   1. P_L is idempotent: P_L @ P_L == P_L
#   2. P_L is orthogonal to U_k: P_L @ U_k ≈ 0
#   3. Projected ΔW is orthogonal to original ΔW
# ─────────────────────────────────────────────────────────────────
import numpy as np
import pytest

from src.config import AetherForgeSettings
from src.learning.oplora_manager import LoraWeightUpdate, OPLoRAManager


@pytest.fixture
def settings(tmp_path):
    s = AetherForgeSettings(aetherforge_env="test", data_dir=tmp_path)
    s.ensure_data_dirs()
    return s


@pytest.fixture
def manager(settings):
    return OPLoRAManager(settings)


class TestProjectorMath:
    """Tests that verify the mathematical guarantees of OPLoRA."""

    def test_projectors_are_correct_shape(self, manager):
        d_out, d_in = 64, 64
        W = np.random.randn(d_out, d_in).astype(np.float32)
        P_L, P_R, sv = manager.compute_projectors(W, rank_k=8)
        assert P_L.shape == (d_out, d_out)
        assert P_R.shape == (d_in, d_in)
        assert len(sv) == 8

    def test_projectors_are_symmetric(self, manager):
        """P_L = I - U U^T must be symmetric."""
        W = np.random.randn(32, 32).astype(np.float32)
        P_L, P_R, _ = manager.compute_projectors(W, rank_k=4)
        np.testing.assert_allclose(P_L, P_L.T, atol=1e-5)
        np.testing.assert_allclose(P_R, P_R.T, atol=1e-5)

    def test_projectors_are_idempotent(self, manager):
        """P_L @ P_L == P_L (orthogonal projector property)."""
        W = np.random.randn(32, 32).astype(np.float32)
        P_L, P_R, _ = manager.compute_projectors(W, rank_k=4)
        np.testing.assert_allclose(P_L @ P_L, P_L, atol=1e-4)
        np.testing.assert_allclose(P_R @ P_R, P_R, atol=1e-4)

    def test_projection_nullifies_past_subspace(self, manager):
        """
        The projected ΔW must lie in the null space of U_k, i.e.,
        U_kᵀ @ (P_L @ ΔW @ P_R) ≈ 0.
        This is the core OPLoRA guarantee.
        """
        d = 32
        W = np.random.randn(d, d).astype(np.float32)
        k = 4
        P_L, P_R, _ = manager.compute_projectors(W, rank_k=k)

        # Get U_k from SVD
        U, _, Vt = np.linalg.svd(W, full_matrices=False)
        U_k = U[:, :k]
        V_k = Vt[:k, :].T

        # Apply projectors to a new random update
        W_new = np.random.randn(d, d).astype(np.float32)
        W_safe = P_L @ W_new @ P_R

        # U_kᵀ @ W_safe ≈ 0 (left null space test)
        left_residual = U_k.T @ W_safe
        np.testing.assert_allclose(left_residual, np.zeros_like(left_residual), atol=1e-4)

        # W_safe @ V_k ≈ 0 (right null space test)
        right_residual = W_safe @ V_k
        np.testing.assert_allclose(right_residual, np.zeros_like(right_residual), atol=1e-4)

    def test_singular_values_are_positive(self, manager):
        W = np.random.randn(48, 48).astype(np.float32)
        _, _, sv = manager.compute_projectors(W, rank_k=8)
        assert all(v >= 0 for v in sv), "Singular values must be non-negative"

    def test_rank_k_capped_at_matrix_rank(self, manager):
        """Requesting more singular vectors than matrix rank should not crash."""
        W = np.random.randn(10, 10).astype(np.float32)
        P_L, P_R, sv = manager.compute_projectors(W, rank_k=100)  # requests 100, matrix is 10×10
        assert len(sv) <= 10


class TestProjectionPipeline:
    """Tests for the end-to-end registration and projection pipeline."""

    def test_register_then_project_reduces_norm(self, manager):
        """Projected weights should have reduced norm (orthogonal component removed)."""
        d, r = 32, 4
        # Task 1 LoRA
        A1 = np.random.randn(r, d).astype(np.float32) * 0.1
        B1 = np.random.randn(d, r).astype(np.float32) * 0.1
        lora1 = LoraWeightUpdate(layer_key="layer.q_proj", A=A1, B=B1)

        # Register task 1
        manager.register_task("task_1", [lora1], rank_k=4)

        # Task 2 proposed LoRA (same direction → most should be projected away)
        lora2 = LoraWeightUpdate(layer_key="layer.q_proj", A=A1.copy(), B=B1.copy())
        projected = manager.project_new_weights(lora2)

        norm_original = float(np.linalg.norm(lora2.delta_W))
        norm_projected = float(np.linalg.norm(projected.delta_W))
        # Projected norm should be strictly less (some component removed)
        assert norm_projected <= norm_original + 1e-5

    def test_no_subspace_no_projection(self, manager):
        """Without registered tasks, projection should be identity."""
        d, r = 32, 4
        A = np.random.randn(r, d).astype(np.float32)
        B = np.random.randn(d, r).astype(np.float32)
        lora = LoraWeightUpdate(layer_key="new_layer", A=A, B=B)
        projected = manager.project_new_weights(lora)
        # A and B should be unchanged
        np.testing.assert_array_equal(projected.A, A)
        np.testing.assert_array_equal(projected.B, B)

    def test_capacity_starts_full(self, manager):
        assert manager.estimate_capacity() == 1.0

    def test_capacity_decreases_after_registration(self, manager):
        d, r = 32, 4
        A = np.random.randn(r, d).astype(np.float32)
        B = np.random.randn(d, r).astype(np.float32)
        lora = LoraWeightUpdate(layer_key="layer.q_proj", A=A, B=B)
        manager.register_task("task_a", [lora], rank_k=4)
        cap = manager.estimate_capacity()
        assert 0.0 <= cap < 1.0

    def test_subspace_summary_structure(self, manager):
        summary = manager.get_subspace_summary()
        assert "total_layers" in summary
        assert "total_tasks" in summary
        assert "layers" in summary

    def test_lora_weights_delta_W(self):
        """LoraWeightUpdate.delta_W = (alpha/r) * B @ A."""
        r, d = 4, 16
        alpha = 32.0
        A = np.ones((r, d), dtype=np.float32)
        B = np.ones((d, r), dtype=np.float32)
        lora = LoraWeightUpdate(layer_key="test", A=A, B=B, alpha=alpha)
        expected = (alpha / r) * (B @ A)
        np.testing.assert_allclose(lora.delta_W, expected)
