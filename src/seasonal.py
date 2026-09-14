"""Phase H8 -- turning several seasonal composites (each already a real
Sentinel-2 composite from src.imagery.get_imagery) into per-pixel seasonal
predictor layers: how often a pixel is water across seasons, and how much
its vegetation greenness swings between the driest and wettest parts of the
year.

Pure numpy, unit-tested with synthetic arrays in tests/test_seasonal.py.
Real multi-season fetching is verified by actually running
notebooks/09_seasonal_dynamics.ipynb against live data.
"""

from __future__ import annotations

import numpy as np


def water_persistence(water_masks: list[np.ndarray]) -> np.ndarray:
    """Fraction of seasons (0.0-1.0) each pixel is classified as water --
    e.g. 1.0 means water in every season given (a permanent water body),
    0.0 means never water, and a value like 0.33 with 3 seasons means
    water in exactly one of them (seasonal/ephemeral water)."""
    if not water_masks:
        raise ValueError("water_masks must contain at least one season")
    return np.mean(np.stack(water_masks).astype("float32"), axis=0)


def seasonal_ndvi_range(ndvi_layers: list[np.ndarray]) -> np.ndarray:
    """Per-pixel max minus min NDVI across seasons -- how much a pixel's
    vegetation greenness actually swings over the year. Near zero for
    permanently bare/built-up/water pixels; large for cropland or
    seasonal vegetation that greens up and dies back."""
    if not ndvi_layers:
        raise ValueError("ndvi_layers must contain at least one season")
    stack = np.stack(ndvi_layers)
    return np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)


def green_up(dry_season_ndvi: np.ndarray, wet_season_ndvi: np.ndarray) -> np.ndarray:
    """Wet-season NDVI minus dry-season NDVI -- a directional signal
    (positive = greened up in the wet season, negative = actually greener
    in the dry season, which is real and happens for e.g. irrigated
    dry-season cropland) rather than seasonal_ndvi_range's undirected
    swing."""
    return wet_season_ndvi - dry_season_ndvi
