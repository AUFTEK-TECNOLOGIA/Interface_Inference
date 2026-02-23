"""Loading and normalizing raw AS7341 spectra."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
import numpy as np

from .sensor import as7341_basic_counts, RAW_BANDS


def load_raw(path: Path) -> dict:
    """Load raw AS7341 JSON file as dictionary."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_block(block: dict, config: dict) -> Dict[str, np.ndarray]:
    """Return all RAW_BANDS normalized as basic_counts."""
    norm = {}
    for band in RAW_BANDS:
        if band in block:
            norm[band] = as7341_basic_counts(block[band], config)
    return norm


def load_channels(block: dict, config: dict, channels: list[str]) -> Dict[str, np.ndarray]:
    """Load selected channels applying basic_counts normalization."""
    norm = normalize_block(block, config)
    return {ch: norm.get(ch, np.array([], float)) for ch in channels}
