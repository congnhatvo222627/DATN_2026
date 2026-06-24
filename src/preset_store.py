"""Preset loading and saving utilities."""

import copy
import json
from pathlib import Path

from .config import (
    DEFAULT_HOUGH_PARAMS,
    DEFAULT_RADIAL_PARAMS,
    DEFAULT_ROI_PARAMS,
    DEFAULT_TAB_EDGE_PARAMS,
    HOUGH_PRESET_PATH,
    RADIAL_PRESET_PATH,
    ROI_PRESET_PATH,
    TAB_EDGE_PRESET_PATH,
)
from .io_utils import ensure_dir


def merge_dict(default, override):
    """Recursively merge override into a copy of default."""
    merged = copy.deepcopy(default)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_preset(path, default):
    """Load a JSON preset and merge missing keys with defaults."""
    path_obj = Path(path)
    if not path_obj.is_file():
        return copy.deepcopy(default)
    with path_obj.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "config" in payload and isinstance(payload["config"], dict):
        payload = payload["config"]
    return merge_dict(default, payload)


def save_preset(path, data):
    """Save a preset as readable JSON."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path_obj


def ensure_default_presets():
    """Create default preset files when missing."""
    preset_map = {
        HOUGH_PRESET_PATH: DEFAULT_HOUGH_PARAMS,
        ROI_PRESET_PATH: DEFAULT_ROI_PARAMS,
        TAB_EDGE_PRESET_PATH: DEFAULT_TAB_EDGE_PARAMS,
        RADIAL_PRESET_PATH: DEFAULT_RADIAL_PARAMS,
    }
    for path, default in preset_map.items():
        if not Path(path).is_file():
            save_preset(path, default)

