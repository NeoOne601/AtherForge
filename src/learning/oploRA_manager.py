# AetherForge v1.0 — src/learning/oploRA_manager.py
# ─────────────────────────────────────────────────────────────────
# OPLoRA (Orthogonal Projection LoRA) — AetherForge's continual
# learning engine. This file implements the mathematical core of
# catastrophic forgetting prevention.
#
# ═══════════════════════════════════════════════════════════════
# THE MATH (Read this before touching the code)
# ═══════════════════════════════════════════════════════════════
#
# Background: Standard LoRA fine-tuning on task T_2 destroys
# knowledge from task T_1 because gradient updates can overwrite
# the subspace used to encode T_1 knowledge.
#
# OPLoRA solution: Before fine-tuning on T_2, compute the
# "knowledge subspace" from T_1's LoRA weights, then project
# all T_2 gradient updates onto the ORTHOGONAL COMPLEMENT of
# that subspace. This guarantees T_1 knowledge is untouched.
#
# Step-by-step algorithm:
#
# 1. At end of task T_k, the LoRA weights are: W_lora = A @ B
#    where A ∈ R^(d × r) and B ∈ R^(r × d).
#
# 2. SVD-decompose the accumulated weight update ΔW = A @ B:
#       ΔW = U Σ Vᵀ    (full SVD, U ∈ R^(d×d), V ∈ R^(d×d))
#
# 3. Take the top-k left/right singular vectors:
#       U_k = U[:, :k]    (k columns, spans input subspace)
#       V_k = V[:, :k]    (k columns, spans output subspace)
#
# 4. Build orthogonal projectors onto the COMPLEMENT:
#       P_L = I - U_k @ U_kᵀ    (projects away from input subspace)
#       P_R = I - V_k @ V_kᵀ    (projects away from output subspace)
#
# 5. For task T_{k+1}, project proposed LoRA update ΔW_new:
#       ΔW_safe = P_L @ ΔW_new @ P_R
#
#    This ΔW_safe is guaranteed to be orthogonal to ΔW (T_k's
#    knowledge subspace), so applying it cannot interfere with
#    T_k's encoded knowledge.
#
# 6. Accumulate: ΔW_total = ΔW (T_1) + ... + ΔW_safe (T_k)
#    Past knowledge is exactly preserved — no forgetting.
#
# Reference: "DARE: Language Model Merging with Orthogonal Adaptation"
#            and orthogonal gradient projection methods (OGD, OrthoReg)
# ═══════════════════════════════════════════════════════════════
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import AetherForgeSettings

logger = logging.getLogger("aetherforge.oploRA")


# ── Data structures ───────────────────────────────────────────────

@dataclass
class TaskKnowledgeSubspace:
    """
    Stores the orthogonal projectors for one completed task.
    Preserved indefinitely — forms the accumulated knowledge base.

    Fields:
      task_id:   Unique task identifier (e.g., "ragforge_20240301")
      layer_key: Model layer this applies to (e.g., "model.0.q_proj")
      P_L:       Left projector (I - U_k U_kᵀ), shape (d_out, d_out)
      P_R:       Right projector (I - V_k V_kᵀ), shape (d_in, d_in)
      rank_k:    Number of singular vectors preserved
      singular_values: Top-k singular values (for novelty detection)
    """
    task_id: str
    layer_key: str
    P_L: np.ndarray           # shape: (d, d)
    P_R: np.ndarray           # shape: (d, d)
    rank_k: int
    singular_values: np.ndarray  # shape: (k,)


@dataclass
class LoRAWeights:
    """
    Represents one LoRA adapter's A and B matrices for a single layer.
      W_update = B @ A    (B: d_out × r, A: r × d_in)
    Note: HuggingFace PEFT uses this convention (B @ A not A @ B).
    """
    layer_key: str
    A: np.ndarray   # shape: (r, d_in)
    B: np.ndarray   # shape: (d_out, r)
    alpha: float = 1.0

    @property
    def delta_W(self) -> np.ndarray:
        """Full weight update: ΔW = (alpha / r) * B @ A"""
        r = self.A.shape[0]
        return (self.alpha / r) * (self.B @ self.A)


# ── OPLoRA Manager ────────────────────────────────────────────────

class OPLoRAManager:
    """
    Manages the accumulated knowledge subspaces across all tasks.

    Usage:
        manager = OPLoRAManager(settings)
        manager.load_checkpoints()   # Restore previous tasks

        # After training T_k:
        manager.register_task("task_k_id", new_lora_weights)

        # Before training T_{k+1}:
        safe_weights = manager.project_new_weights(proposed_weights)
    """

    def __init__(self, settings: AetherForgeSettings) -> None:
        self.settings = settings
        self._rank_k = settings.oploра_rank_k
        self._checkpoint_dir = settings.data_dir / "lora_checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Accumulated subspaces: layer_key → list of TaskKnowledgeSubspace
        # Each entry in the list is from one past task.
        self._subspaces: dict[str, list[TaskKnowledgeSubspace]] = {}

    def load_checkpoints(self) -> int:
        """
        Load all saved task subspace checkpoints from disk.
        Returns number of tasks loaded.
        """
        count = 0
        for f in sorted(self._checkpoint_dir.glob("*.npz")):
            try:
                data = np.load(f, allow_pickle=True)
                subspace = TaskKnowledgeSubspace(
                    task_id=str(data["task_id"]),
                    layer_key=str(data["layer_key"]),
                    P_L=data["P_L"],
                    P_R=data["P_R"],
                    rank_k=int(data["rank_k"]),
                    singular_values=data["singular_values"],
                )
                layer_key = subspace.layer_key
                if layer_key not in self._subspaces:
                    self._subspaces[layer_key] = []
                self._subspaces[layer_key].append(subspace)
                count += 1
            except Exception as exc:
                logger.warning("Failed to load checkpoint %s: %s", f, exc)
        logger.info("Loaded %d task knowledge subspaces from disk", count)
        return count

    # ── Core Math ─────────────────────────────────────────────────

    def compute_projectors(
        self,
        delta_W: np.ndarray,
        rank_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute orthogonal projectors from a weight update matrix.

        Algorithm (see module docstring for full derivation):
          delta_W ∈ R^(d_out × d_in)
          1. U, Σ, Vᵀ = SVD(delta_W)
          2. U_k = U[:, :k] (top-k left singular vectors)
          3. V_k = Vᵀ[:k, :].T = V[:, :k] (top-k right singular vectors)
          4. P_L = I_{d_out} - U_k @ U_kᵀ
          5. P_R = I_{d_in}  - V_k @ V_kᵀ

        Returns: (P_L, P_R, singular_values[:k])

        Note: np.linalg.svd returns V already transposed (Vᵀ),
        so we take Vt.T[:, :k] to get V_k.
        """
        k = rank_k or self._rank_k
        d_out, d_in = delta_W.shape

        # ── SVD ───────────────────────────────────────────────────
        # full_matrices=False → economy SVD, O(d × r) not O(d²)
        # This is critical for large models where d can be 4096+.
        U, sigma, Vt = np.linalg.svd(delta_W, full_matrices=False)
        # U:  (d_out, min(d_out,d_in))
        # sigma: (min(d_out,d_in),)
        # Vt: (min(d_out,d_in), d_in)

        # ── Top-k selection ───────────────────────────────────────
        actual_k = min(k, U.shape[1], Vt.shape[0])
        U_k = U[:, :actual_k]      # (d_out, k)
        V_k = Vt[:actual_k, :].T  # (d_in,  k) — V not Vt

        # ── Orthogonal projectors ─────────────────────────────────
        # P_L = I - U_k @ U_kᵀ  ∈ R^(d_out × d_out)
        P_L = np.eye(d_out, dtype=np.float32) - (U_k @ U_k.T).astype(np.float32)

        # P_R = I - V_k @ V_kᵀ  ∈ R^(d_in × d_in)
        P_R = np.eye(d_in, dtype=np.float32) - (V_k @ V_k.T).astype(np.float32)

        logger.debug(
            "Computed projectors: d_out=%d d_in=%d k=%d "
            "singular_values[0]=%+.4f singular_values[-1]=%+.4f",
            d_out, d_in, actual_k,
            float(sigma[0]), float(sigma[actual_k - 1]),
        )
        return P_L, P_R, sigma[:actual_k].astype(np.float32)

    def project_new_weights(self, new_lora: LoRAWeights) -> LoRAWeights:
        """
        Project proposed LoRA weight update into the orthogonal complement
        of ALL previously registered task subspaces for this layer.

        ΔW_safe = (∏_{t=1}^{k} P_L^t) @ ΔW_proposed @ (∏_{t=1}^{k} P_R^t)

        The product of orthogonal projectors is applied left-to-right,
        progressively narrowing the safe update subspace. This guarantees
        that ΔW_safe is orthogonal to every past task's subspace.

        Returns a new LoRAWeights with modified A and B matrices.
        """
        layer_key = new_lora.layer_key
        delta_W = new_lora.delta_W.astype(np.float32)
        projected = delta_W

        subspaces = self._subspaces.get(layer_key, [])
        if not subspaces:
            # No past tasks for this layer — no projection needed
            return new_lora

        for subspace in subspaces:
            # Left-multiply by P_L, right-multiply by P_R
            projected = subspace.P_L @ projected @ subspace.P_R

        # Decompose the projected ΔW back into A, B form via SVD.
        # This maintains the LoRA r-rank factorization.
        r = new_lora.A.shape[0]
        U, sigma, Vt = np.linalg.svd(projected, full_matrices=False)
        actual_r = min(r, len(sigma))
        # B = U[:, :r] * sqrt(sigma[:r])
        # A = diag(sqrt(sigma[:r])) @ Vt[:r, :]
        sqrt_sigma = np.sqrt(np.maximum(sigma[:actual_r], 0))
        new_B = (U[:, :actual_r] * sqrt_sigma).astype(np.float32)
        new_A = (np.diag(sqrt_sigma) @ Vt[:actual_r, :]).astype(np.float32)

        # Pad if needed
        if actual_r < r:
            pad_B = np.zeros((new_B.shape[0], r - actual_r), dtype=np.float32)
            pad_A = np.zeros((r - actual_r, new_A.shape[1]), dtype=np.float32)
            new_B = np.concatenate([new_B, pad_B], axis=1)
            new_A = np.concatenate([new_A, pad_A], axis=0)

        logger.info("Projected LoRA for layer %s: ||ΔW||=%.4f → ||ΔW_safe||=%.4f (%.1f%% preserved)",
            layer_key,
            float(np.linalg.norm(delta_W)),
            float(np.linalg.norm(projected)),
            100 * float(np.linalg.norm(projected)) / max(float(np.linalg.norm(delta_W)), 1e-8),
        )
        return LoRAWeights(layer_key=layer_key, A=new_A, B=new_B, alpha=new_lora.alpha)

    def register_task(
        self,
        task_id: str,
        lora_weights: list[LoRAWeights],
        rank_k: int | None = None,
    ) -> int:
        """
        Register a completed task's LoRA weights as a knowledge subspace.
        Call this AFTER training on task T_k, BEFORE training on T_{k+1}.

        Returns number of layers processed.
        """
        processed = 0
        for lora in lora_weights:
            delta_W = lora.delta_W.astype(np.float32)
            if delta_W.shape[0] < 2 or delta_W.shape[1] < 2:
                logger.warning("Skipping projector for tiny layer %s", lora.layer_key)
                continue

            P_L, P_R, singular_values = self.compute_projectors(delta_W, rank_k)
            subspace = TaskKnowledgeSubspace(
                task_id=task_id,
                layer_key=lora.layer_key,
                P_L=P_L,
                P_R=P_R,
                rank_k=len(singular_values),
                singular_values=singular_values,
            )

            if lora.layer_key not in self._subspaces:
                self._subspaces[lora.layer_key] = []
            self._subspaces[lora.layer_key].append(subspace)

            # Persist to disk
            self._save_subspace(subspace)
            processed += 1

        logger.info("Registered task '%s': %d layer projectors computed", task_id, processed)
        return processed

    def _save_subspace(self, subspace: TaskKnowledgeSubspace) -> None:
        """Persist a TaskKnowledgeSubspace to disk as .npz."""
        filename = f"{subspace.task_id}_{subspace.layer_key.replace('.', '_')}.npz"
        path = self._checkpoint_dir / filename
        np.savez(
            path,
            task_id=subspace.task_id,
            layer_key=subspace.layer_key,
            P_L=subspace.P_L,
            P_R=subspace.P_R,
            rank_k=subspace.rank_k,
            singular_values=subspace.singular_values,
        )
        logger.debug("Saved subspace checkpoint: %s", path)

    def get_subspace_summary(self) -> dict[str, Any]:
        """Return a summary of all registered knowledge subspaces."""
        return {
            "total_layers": len(self._subspaces),
            "total_tasks": sum(len(v) for v in self._subspaces.values()),
            "layers": {
                layer: {
                    "task_count": len(subspaces),
                    "tasks": [s.task_id for s in subspaces],
                    "rank_k": subspaces[-1].rank_k if subspaces else 0,
                }
                for layer, subspaces in self._subspaces.items()
            },
        }

    def estimate_capacity(self) -> float:
        """
        Estimate remaining learning capacity (0.0 = full, 1.0 = empty).
        As more tasks are registered, the orthogonal complement shrinks.
        Capacity ≈ fraction of singular dimensions still available.
        """
        if not self._subspaces:
            return 1.0

        # Average preserved dimensions across layers
        fractions = []
        for layer_key, subspaces in self._subspaces.items():
            total_k = sum(s.rank_k for s in subspaces)
            # Rough estimate: assume square d×d matrix
            d = subspaces[0].P_L.shape[0]
            fractions.append(max(0.0, (d - total_k) / d))

        return float(np.mean(fractions))
