"""Phase H2 -- turning ESA WorldCover's 11 detailed classes into the 4
target classes this project's plan actually asks for (water / built-up /
vegetation / bare), and a spatial (block, not random-pixel) train/test split
for evaluating the RandomForest baseline honestly.

Pure logic here is unit-tested with synthetic arrays in
tests/test_landcover.py. The real WorldCover fetch + RF training is I/O and
model-fitting, verified instead by actually running
notebooks/03_landcover_baseline.ipynb against live data.
"""

from __future__ import annotations

import numpy as np

WATER, BUILT_UP, VEGETATION, BARE = 0, 1, 2, 3
CLASS_NAMES = ["water", "built_up", "vegetation", "bare"]

# ESA WorldCover v200 class codes -> this project's 4 target classes.
# 90 (herbaceous wetland) is mapped to WATER, not VEGETATION: for a
# flood/dengue risk-mapping use case, a wetland's hydrological role (wet,
# standing-water-adjacent) matters more than its plant cover. This is a
# judgment call, not a fact -- documented so a later phase can revisit it.
WORLDCOVER_TO_TARGET_CLASS = {
    10: VEGETATION,  # tree cover
    20: VEGETATION,  # shrubland
    30: VEGETATION,  # grassland
    40: VEGETATION,  # cropland
    50: BUILT_UP,    # built-up
    60: BARE,        # bare / sparse vegetation
    70: BARE,        # snow / ice (not expected in Bangladesh, included for completeness)
    80: WATER,       # water bodies
    90: WATER,       # herbaceous wetland -- see note above
    95: VEGETATION,  # mangroves
    100: BARE,       # moss / lichen
}


def reclassify_worldcover(worldcover: np.ndarray) -> np.ndarray:
    """Maps raw ESA WorldCover class codes to this project's 4 target
    classes. Any code not in the mapping (unexpected input) becomes -1
    ("unknown"), not a silent guess -- callers should mask those out rather
    than train/evaluate on them."""
    result = np.full(worldcover.shape, -1, dtype="int8")
    for wc_code, target_class in WORLDCOVER_TO_TARGET_CLASS.items():
        result[worldcover == wc_code] = target_class
    return result


def build_feature_stack(green: np.ndarray, red: np.ndarray, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Stacks the 4 raw bands plus the 3 spectral indices into a
    (7, height, width) feature cube for pixel-wise classification --
    the raw reflectance values carry information the normalized indices
    compress away (e.g. absolute brightness), so both are included."""
    from src.indices import ndbi, ndvi, ndwi

    features = np.stack([
        green, red, nir, swir,
        ndvi(nir, red), ndwi(green, nir), ndbi(swir, nir),
    ])
    return features.astype("float32")


def spatial_block_split(height: int, width: int, test_fraction: float = 0.3, axis: str = "x") -> np.ndarray:
    """A contiguous spatial block held out for testing -- e.g. the
    rightmost `test_fraction` of columns -- rather than a random pixel
    split. Adjacent pixels are highly spatially autocorrelated (a
    neighborhood is almost always the same land-cover class), so a random
    per-pixel split leaks test information into training and makes accuracy
    look far better than it would on a genuinely unseen area. Returns a
    boolean mask, True = held out for testing.

    axis='x' splits by column (a vertical line dividing the AOI east/west);
    axis='y' splits by row (north/south)."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")
    mask = np.zeros((height, width), dtype=bool)
    if axis == "x":
        split_col = int(width * (1 - test_fraction))
        mask[:, split_col:] = True
    elif axis == "y":
        split_row = int(height * (1 - test_fraction))
        mask[split_row:, :] = True
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return mask
