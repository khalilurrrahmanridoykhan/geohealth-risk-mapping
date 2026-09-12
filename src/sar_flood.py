"""Phase H5 -- SAR-based water/flood detection: water = low backscatter
(a smooth water surface reflects radar away from the sensor instead of
back to it, unlike rough vegetation/built-up surfaces).

Pure numpy, unit-tested with synthetic arrays in tests/test_sar_flood.py.
Real Sentinel-1 fetching is src.imagery.get_imagery(sensor='sentinel-1');
demonstrated end to end against a real before/during-flood pair in
notebooks/06_sar_flood_mapping.ipynb.
"""

from __future__ import annotations

import numpy as np


def db_from_power(power: np.ndarray) -> np.ndarray:
    """Linear power (gamma0/sigma0, as sentinel-1-rtc provides) -> decibels.
    Non-positive values (nodata, sensor artifacts) become NaN rather than
    -inf or a runtime warning silently propagating downstream."""
    power = power.astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(power > 0, power, np.nan))
    return db.astype("float32")


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's method: the threshold that minimizes within-class variance
    for a bimodal distribution (here, water vs. land backscatter) --
    implemented directly rather than pulling in scikit-image for one
    function. NaNs are dropped before histogramming."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite values to threshold")
    hist, bin_edges = np.histogram(finite, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight_total = hist.sum()

    weight_below = np.cumsum(hist)
    weight_above = weight_total - weight_below
    # Avoid division by zero at the histogram's edges (bins with no data on one side).
    safe_below = np.where(weight_below == 0, 1, weight_below)
    safe_above = np.where(weight_above == 0, 1, weight_above)

    mean_below = np.cumsum(hist * bin_centers) / safe_below
    total_mean = np.sum(hist * bin_centers)
    mean_above = (total_mean - np.cumsum(hist * bin_centers)) / safe_above

    between_class_variance = weight_below * weight_above * (mean_below - mean_above) ** 2
    best_bin = np.argmax(between_class_variance)
    return float(bin_centers[best_bin])


def water_mask_from_backscatter(vv_db: np.ndarray, threshold_db: float) -> np.ndarray:
    """True where backscatter is below the threshold -- i.e. water/smooth
    surface. NaN input pixels are never classified as water (False)."""
    return np.where(np.isnan(vv_db), False, vv_db < threshold_db)


def flood_extent(before_water: np.ndarray, during_water: np.ndarray) -> np.ndarray:
    """Newly flooded pixels: water now, but not before -- real change
    detection, not just the during-event water mask on its own (which
    would also include rivers/ponds that are wet in every season)."""
    return during_water & ~before_water
