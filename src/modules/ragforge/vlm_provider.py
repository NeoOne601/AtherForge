"""
Hardware-Adaptive VLM Provider System.
Allows dynamic selection of Vision-Language Models based on hardware constraints.
Models are stored on external storage by default to preserve internal SSD space.
"""

import asyncio
import gc
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import structlog

from src.config import get_settings

logger = structlog.get_logger("aetherforge.ragforge.vlm_provider")

# Allow users to put models on an external drive to save SSD space
# Default to ~/.cache/atherforge/models if not specified
_DEFAULT_CACHE = Path.home() / ".cache" / "atherforge" / "models"
MODEL_DIR = Path(os.getenv("ATHERFORGE_MODEL_DIR", str(_DEFAULT_CACHE)))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODEL_DIR / "huggingface")

# ── Provider Singleton Registry ───────────────────────────────────
# Prevents factory instantiation from running torch.backends.mps.is_available()
# (which loads libTorch) on every API call.
_PROVIDER_REGISTRY: dict[str, "VLMProvider"] = {}  # id -> instance


class VLMProvider(ABC):
    """Abstract base class for hardware-adaptive VLM providers."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier (e.g., 'smolvlm', 'florence2')."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        pass

    @property
    @abstractmethod
    def required_ram_gb(self) -> float:
        """Estimated inference RAM in GB."""
        pass

    @property
    @abstractmethod
    def tier(self) -> str:
        """Tier designation: Lite, Balanced, or Pro."""
        pass

    @abstractmethod
    async def load_model(self) -> None:
        """Load model into memory. Download if necessary."""
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload model and aggressively free memory."""
        pass

    @abstractmethod
    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        """Process an image and return extracted text/analysis."""
        pass


class SmolVLMProvider(VLMProvider):
    """
    Tier 1: Lite (SmolVLM-256M)
    Best for 8GB devices. <1GB memory footprint.
    Strong at document OCR and basic chart reading.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "mps" if self._check_mps() else "cpu"

    def _check_mps(self):
        import torch

        return torch.backends.mps.is_available()

    @property
    def id(self) -> str:
        return "smolvlm-256m"

    @property
    def name(self) -> str:
        return "SmolVLM 256M (Lite)"

    @property
    def required_ram_gb(self) -> float:
        return 0.8

    @property
    def tier(self) -> str:
        return "Lite"

    async def load_model(self) -> None:
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info("Loading %s to %s...", self.name, self.device)
        repo_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

        # Load asynchronously so we don't block the event loop
        def _load():
            self.processor = AutoProcessor.from_pretrained(repo_id)
            self.model = AutoModelForVision2Seq.from_pretrained(
                repo_id,
                torch_dtype=torch.bfloat16,
                _attn_implementation="eager",  # Flash attention often fails on MPS
            ).to(self.device)

        await asyncio.to_thread(_load)
        logger.info("%s loaded successfully.", self.name)

    async def unload_model(self) -> None:
        if self.model is not None:
            logger.info("Unloading %s to free memory...", self.name)
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            try:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        if self.model is None:
            await self.load_model()

        import io

        from PIL import Image

        def _analyze():
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            ]

            # Prepare inputs
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt")
            inputs = inputs.to(self.device)

            # Generate
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
            )

            # Decode only the newly generated tokens
            generated_texts = self.processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return generated_texts[0].strip()

        return await asyncio.to_thread(_analyze)


class FlorenceProvider(VLMProvider):
    """
    Tier 2: Balanced (Florence-2-base)
    Best for strong OCR, table extraction, and chart labels.
    ~1GB memory footprint. Extremely fast on edge.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "mps" if self._check_mps() else "cpu"

    def _check_mps(self):
        import torch

        return torch.backends.mps.is_available()

    @property
    def id(self) -> str:
        return "florence2-base"

    @property
    def name(self) -> str:
        return "Florence-2 Base (Balanced)"

    @property
    def required_ram_gb(self) -> float:
        return 1.0

    @property
    def tier(self) -> str:
        return "Balanced"

    async def load_model(self) -> None:
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info("Loading %s to %s...", self.name, self.device)
        repo_id = "microsoft/Florence-2-base"

        def _load():
            self.processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                trust_remote_code=True,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            ).to(self.device)

        await asyncio.to_thread(_load)
        logger.info("%s loaded successfully.", self.name)

    async def unload_model(self) -> None:
        if self.model is not None:
            logger.info("Unloading %s...", self.name)
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            try:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        if self.model is None:
            await self.load_model()

        import io

        from PIL import Image

        def _analyze():
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Always use <MORE_DETAILED_CAPTION> for RAGForge visual analysis.
            # Florence-2 <CAPTION> mode only produces ~10 words — useless for charts.
            # <MORE_DETAILED_CAPTION> gives 3-4 sentences with layout and content.
            # For charts/tables, <DENSE_REGION_CAPTION> would be even better but
            # requires bounding boxes; use MORE_DETAILED_CAPTION as the safe default.
            task_prompt = "<MORE_DETAILED_CAPTION>"

            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(
                self.device, self.model.dtype
            )

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=500,
                num_beams=3,
            )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[
                0
            ]
            parsed_answer = self.processor.post_process_generation(
                generated_text, task=task_prompt, image_size=(image.width, image.height)
            )
            return parsed_answer.get(task_prompt, generated_text)

        return await asyncio.to_thread(_analyze)


# ── Apple VLM Provider (MLX-VLM) ──────────────────────────────────


class AppleVLMProvider(VLMProvider):
    """
    Tier 3: Apple Optimized (Apple-VLM)
    Optimized for Apple Silicon using mlx-vlm.
    Loads from external drive by default.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "mps"  # Always mps for Apple VLM
        self.settings = get_settings()

    @property
    def id(self) -> str:
        return "apple-vlm"

    @property
    def name(self) -> str:
        return "Apple VLM (Optimized)"

    @property
    def required_ram_gb(self) -> float:
        return 4.0

    @property
    def tier(self) -> str:
        return "Optimized (Apple)"

    async def load_model(self) -> None:
        if self.model is not None:
            return

        try:
            import mlx_vlm
            from mlx_vlm.utils import load
        except ImportError:
            logger.error("mlx-vlm not installed. Apple VLM requires 'pip install mlx-vlm'")
            return

        logger.info("Loading %s from %s...", self.name, self.settings.apple_vlm_model_path)

        def _load():
            # mlx-vlm.utils.load returns (model, processor)
            self.model, self.processor = load(str(self.settings.apple_vlm_model_path))

        await asyncio.to_thread(_load)
        logger.info("%s loaded successfully.", self.name)

    async def unload_model(self) -> None:
        if self.model is not None:
            logger.info("Unloading %s...", self.name)
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            # MLX manages its own memory, but gc.collect() helps Python

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        if self.model is None:
            await self.load_model()
            if self.model is None:
                return "ERROR: Apple VLM not available (mlx-vlm missing or model not found)"

        import io

        from PIL import Image

        def _analyze():
            try:
                from mlx_vlm.utils import generate

                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Generate using mlx-vlm
                response = generate(self.model, self.processor, image, prompt, verbose=False)
                return response.strip()
            except Exception as e:
                logger.error("Apple VLM analysis failed: %s", e)
                return f"ERROR: Apple VLM failed: {str(e)}"

        return await asyncio.to_thread(_analyze)


# QwenVLProvider removed — requires gated HuggingFace token and CUDA GPTQ.
# Use OllamaQwenProvider (Tier 3 / Ultra) instead: fully offline, no token,
# runs qwen3.5:9b via a separate Ollama process (no Python RAM overhead).


class OllamaQwenProvider(VLMProvider):
    """
    Tier 4: Ultra
    Queries a local Ollama instance for qwen3.5:9b or similar models.
    """

    def __init__(self, model_name: str = "qwen3.5:9b"):
        self._model_name = model_name  # Honor the parameter
        self.device = "mps"  # Handled externally by Ollama

    @property
    def id(self) -> str:
        return "ollama-qwen3.5-9b"

    @property
    def name(self) -> str:
        return "Qwen 3.5 9B (Ollama)"

    @property
    def tier(self) -> str:
        return "Ultra (Ollama)"

    @property
    def required_ram_gb(self) -> float:
        return 6.0  # Approx memory required

    async def load_model(self):
        import asyncio
        import json
        import urllib.request

        def _pull():
            req = urllib.request.Request(
                "http://localhost:11434/api/pull",
                data=json.dumps({"model": self._model_name}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req) as res:
                    pass
                logger.info("Ollama model %s pulled successfully.", self._model_name)
            except Exception as e:
                logger.warning(
                    "Ollama pull failed for %s (is Ollama running?): %s", self._model_name, e
                )

        await asyncio.to_thread(_pull)

    async def unload_model(self):
        import asyncio
        import json
        import urllib.request

        def _unload():
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({"model": self._model_name, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req)
                logger.info("Ollama model %s unloaded.", self._model_name)
            except Exception:
                pass

        await asyncio.to_thread(_unload)

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        import asyncio
        import base64
        import json
        import urllib.request

        def _generate():
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps(
                    {
                        "model": self._model_name,
                        "prompt": prompt,
                        "images": [b64_image],
                        "stream": False,
                        "keep_alive": "5m",
                        "options": {"num_ctx": 4096},
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req) as response:
                    res_body = response.read()
                    data = json.loads(res_body)
                    return data.get("response", "")
            except Exception as e:
                logger.error("Ollama generation failed: %s", e)
                return "NO_FIGURES (Ollama Error)"

        return await asyncio.to_thread(_generate)


class OllamaQwenVL2BProvider(OllamaQwenProvider):
    """
    Highly optimized 4-bit vision-language model.
    Runs via Ollama. Needs ~2.0GB VRAM.
    """

    def __init__(self):
        self._model_name = "qwen2.5vl:3b"

    @property
    def id(self) -> str:
        return "ollama-qwen2.5vl-3b"

    @property
    def name(self) -> str:
        return "Qwen 2.5VL 3B (Ollama)"

    @property
    def tier(self) -> str:
        return "Optimized (Ollama)"

    @property
    def required_ram_gb(self) -> float:
        return 3.5


# Three provider tiers (no gated-token providers):
#   Lite    — SmolVLM 256M  (HuggingFace, in-process, ~0.5GB)
#   Standard — Florence-2   (HuggingFace, in-process, ~1.5GB)
#   Optimized — Qwen 2.5-VL 2B (Ollama, out-of-process, ~2.0GB)
#   Ultra   — Qwen 3.5 9B  (Ollama, out-of-process, 6.6GB, not recommended for 8GB RAM)
AVAILABLE_PROVIDERS = [
    SmolVLMProvider,
    FlorenceProvider,
    AppleVLMProvider,
    OllamaQwenVL2BProvider,
    OllamaQwenProvider,
]


def get_vlm_provider(provider_id: str) -> Optional["VLMProvider"]:
    """Factory to get a VLM provider instance by its ID.

    Uses a lazy singleton registry — each provider class is instantiated
    exactly once and reused on subsequent calls, preventing repeated
    torch library initialization overhead.
    """
    global _PROVIDER_REGISTRY
    if provider_id in _PROVIDER_REGISTRY:
        return _PROVIDER_REGISTRY[provider_id]

    for provider_cls in AVAILABLE_PROVIDERS:
        instance = provider_cls()
        if instance.id == provider_id:
            _PROVIDER_REGISTRY[provider_id] = instance
            return instance
    return None


def list_providers() -> list["VLMProvider"]:
    """Return one singleton instance of each available provider."""
    results = []
    for cls in AVAILABLE_PROVIDERS:
        tmp = cls()
        p = get_vlm_provider(tmp.id)
        if p is not None:
            results.append(p)
    return results
