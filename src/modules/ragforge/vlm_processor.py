# AetherForge v1.0 — src/modules/ragforge/vlm_processor.py
# ─────────────────────────────────────────────────────────────────
# Academic Visual Language Model (VLM) Processor
#
# Ported and adapted from NeoOne601/Ventro (MIT licence).
# Original: infrastructure/cv/vlm_processor.py (financial extraction)
# Adaptation: academic/research paper extraction with:
#   - multi-column layout detection
#   - equation / formula recognition (returns LaTeX)
#   - table extraction with row/column structure
#   - section heading detection
#   - figure caption linking
#
# Uses Qwen2-VL-7B via local Ollama. Falls back gracefully.
# Zero internet required. All processing is on-device.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger("aetherforge.ragforge.vlm")

# Supported VLM models (all run via Ollama locally)
SUPPORTED_VLM_MODELS = {
    "llama3.2-vision": "llama3.2-vision",   # Best M1-optimized edge VLM
    "internvl2":  "internvl2:8b",            # Strong document QA
    "llava":      "llava:13b",               # General vision fallback
    "minicpm-v":  "minicpm-v:latest",        # Lightweight Apple Silicon
}

DEFAULT_VLM_MODEL = "llama3.2-vision"

ACADEMIC_EXTRACTION_PROMPT = """\
You are a research paper analyst. Carefully analyze this page from a scientific or academic document.

Extract ALL content and return ONLY valid JSON with exactly this structure:
{
  "section_heading": "<the section heading visible on this page, or null>",
  "text": "<all readable body text on this page, preserving paragraph structure>",
  "tables": [
    "<table 1 content as markdown table format>",
    "<table 2 content as markdown table format>"
  ],
  "equations": [
    "<equation 1 in LaTeX notation, e.g. E = mc^2>",
    "<equation 2 in LaTeX notation>"
  ],
  "figure_captions": [
    "<caption of figure 1>",
    "<caption of figure 2>"
  ],
  "is_multi_column": <true if the page uses 2 or more columns, else false>,
  "language": "<ISO 639-1 language code, e.g. en, de, zh>",
  "extraction_confidence": <0.0 to 1.0>
}

Rules:
- For tables: use markdown table format (| col1 | col2 |)
- For equations: use LaTeX notation wherever possible
- If a field has no content, use null for strings or [] for arrays
- NEVER include text outside the JSON block
- Reconstruct multi-column text in correct reading order (left-to-right first, then next row)
"""


class AcademicVLMProcessor:
    """
    Visual Language Model processor for extracting structured content
    from complex academic/research document pages.

    Designed for:
      - Multi-column IEEE/ACM/Nature-style research papers
      - Scanned PDFs with handwritten annotations
      - Documents with complex mathematical notation
      - Any language (Qwen2-VL supports 50+ languages)

    Uses Ollama as the local inference backend — no internet required.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = DEFAULT_VLM_MODEL,
        timeout: int = 180,
    ) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        logger.info("AcademicVLMProcessor initialized | model=%s", model)

    def _encode_image(self, image_bytes: bytes) -> str:
        """Base64-encode page image bytes for Ollama API."""
        return base64.b64encode(image_bytes).decode("utf-8")

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Robustly extract JSON from VLM response (handles markdown fences)."""
        # Strip markdown fences
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`")
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1:
            raise ValueError(f"No JSON found in VLM response: {text[:400]}")
        return json.loads(text[start:end])

    async def extract_academic_content(
        self,
        page_image_bytes: bytes,
        language_hint: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a rendered PDF page image to the VLM and extract structured
        academic content: body text, tables, equations, figure captions.

        Returns:
          {
            "section_heading": str | None,
            "text": str,
            "tables": list[str],
            "equations": list[str],
            "figure_captions": list[str],
            "is_multi_column": bool,
            "language": str,
            "extraction_confidence": float,
          }
        """
        prompt = ACADEMIC_EXTRACTION_PROMPT
        if language_hint:
            prompt += f"\n\nNote: The document is likely in language: {language_hint}. Extract accordingly."

        empty_result: dict[str, Any] = {
            "section_heading": None,
            "text": "",
            "tables": [],
            "equations": [],
            "figure_captions": [],
            "is_multi_column": False,
            "language": language_hint or "en",
            "extraction_confidence": 0.0,
        }

        try:
            resp = await self._client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [self._encode_image(page_image_bytes)],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,    # deterministic extraction
                        "num_predict": 4096,   # allow large academic pages
                    },
                },
            )
            resp.raise_for_status()
            response_text = resp.json().get("response", "")
            result = self._parse_json_response(response_text)

            logger.info(
                "VLM page extracted | confidence=%.2f | tables=%d | equations=%d | multi_col=%s",
                result.get("extraction_confidence", 0),
                len(result.get("tables", [])),
                len(result.get("equations", [])),
                result.get("is_multi_column", False),
            )
            # Merge with defaults to guarantee all keys present
            return {**empty_result, **result}

        except httpx.ConnectError:
            logger.warning(
                "VLM unavailable (Ollama not running). "
                "Pull model with: ollama pull %s", self.model
            )
            return empty_result
        except json.JSONDecodeError as e:
            logger.error("VLM JSON parse failed: %s", e)
            return empty_result
        except Exception as e:
            logger.error("VLM extraction error: %s", e)
            return empty_result

    async def detect_language(self, page_image_bytes: bytes) -> str:
        """
        Quick language detection on the first page.
        Returns ISO 639-1 code ('en', 'de', 'zh', 'ja', 'ar', etc.)
        """
        prompt = (
            "Look at this document page. What written language is it in? "
            "Reply with ONLY the ISO 639-1 two-letter code (e.g. en, de, zh, ja, ar, hi, ru). "
            "Nothing else — no explanation, no punctuation."
        )
        try:
            resp = await self._client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [self._encode_image(page_image_bytes)],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 5},
                },
            )
            resp.raise_for_status()
            lang = resp.json().get("response", "en").strip().lower()[:2]
            return lang if re.match(r"^[a-z]{2}$", lang) else "en"
        except Exception:
            return "en"

    async def health_check(self) -> dict[str, Any]:
        """Check if the VLM model is available in the local Ollama instance."""
        try:
            resp = await self._client.get(f"{self.ollama_base_url}/api/tags")
            available_models = [m["name"] for m in resp.json().get("models", [])]
            model_present = any(self.model in m for m in available_models)
            return {
                "ollama_running": True,
                "model_available": model_present,
                "model": self.model,
                "available_models": available_models,
                "hint": f"Run: ollama pull {self.model}" if not model_present else "Ready",
            }
        except Exception as e:
            return {
                "ollama_running": False,
                "model_available": False,
                "model": self.model,
                "error": str(e),
                "hint": "Start Ollama first: ollama serve",
            }

    async def close(self) -> None:
        await self._client.aclose()
