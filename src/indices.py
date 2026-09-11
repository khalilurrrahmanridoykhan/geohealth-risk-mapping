"""Spectral index math shared across phases -- pure numpy, no I/O, so it can
be unit-tested with small synthetic arrays instead of a real scene. All three
follow the same normalized-difference shape but read different bands, which
is why they're kept in one place rather than copy-pasted per notebook.
"""

from __future__ import annotations

import numpy as np


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype("float32")
    b = b.astype("float32")
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom != 0, (a - b) / denom, 0.0)
    return result.astype("float32")


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Vegetation index. High (near +1) = dense healthy vegetation,
    near 0 or negative = bare soil, water, or built-up."""
    return _normalized_difference(nir, red)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters' water index. High (near +1) = open water; land is
    typically negative."""
    return _normalized_difference(green, nir)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Built-up index. High (near +1) = built-up/impervious surface;
    vegetation and water are typically negative."""
    return _normalized_difference(swir, nir)
