#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import iio as _iio
except ImportError:
    _iio = None


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if _iio is not None:
        try:
            return np.asarray(_iio.read(path))
        except Exception:
            pass

    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return np.asarray(Image.open(path))


def save_uint8_image(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)
