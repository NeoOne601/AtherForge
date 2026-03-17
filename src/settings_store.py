from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_FILE = Path("data/settings.json")


def load_saved_settings() -> dict[str, Any]:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_partial_settings(values: dict[str, Any]) -> dict[str, Any]:
    saved = load_saved_settings()
    saved.update(values)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as handle:
        json.dump(saved, handle, indent=2)
    return saved
