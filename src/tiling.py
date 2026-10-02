"""Phase H3 -- chopping a large raster into fixed-size tiles for U-Net
training, and assigning each tile to train/test by spatial position (not
randomly) so the held-out set is a real contiguous region the model never
saw, not just unseen pixels scattered through training tiles.

Pure logic, unit-tested with synthetic arrays in tests/test_tiling.py.
"""

from __future__ import annotations

import numpy as np


def tile_array(array: np.ndarray, tile_size: int = 256) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Splits a (..., H, W) array into non-overlapping tile_size x tile_size
    tiles. Incomplete tiles at the bottom/right edge are dropped rather than
    padded -- a padded tile would train the model on fabricated border
    pixels, which is worse than just having slightly less data. Returns
    (tiles, top_left_coords) with coords as (row_start, col_start) in the
    original array, so a tile's spatial position can be recovered later
    (e.g. for train/test assignment)."""
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    height, width = array.shape[-2], array.shape[-1]
    tiles: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    for row in range(0, height - tile_size + 1, tile_size):
        for col in range(0, width - tile_size + 1, tile_size):
            tiles.append(array[..., row:row + tile_size, col:col + tile_size])
            coords.append((row, col))
    return tiles, coords


def assign_tile_split(coords: list[tuple[int, int]], width: int, test_fraction: float = 0.3) -> list[bool]:
    """True for a tile that falls in the eastern (rightmost) test_fraction
    of the original array's width -- the same spatial-block convention used
    by src.landcover.spatial_block_split, applied per-tile instead of
    per-pixel. A tile's own column start (not its center) decides the
    split, so no tile straddles the boundary and leaks pixels across it."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")
    split_col = width * (1 - test_fraction)
    return [col_start >= split_col for _, col_start in coords]


def untile_array(
    tiles: list[np.ndarray], coords: list[tuple[int, int]], output_shape: tuple[int, int], fill_value=False
) -> np.ndarray:
    """Inverse of tile_array for 2D tiles: stitches tiles back at their
    (row_start, col_start) coords into an array of output_shape. Tiles are
    written in the order given, so later tiles overwrite earlier ones at
    any overlap (tile_array itself never produces overlap; this only
    matters if coords come from elsewhere). Pixels no tile covers -- the
    dropped-incomplete-edge rows/columns tile_array leaves out -- keep
    fill_value (default False, for boolean masks)."""
    out = np.full(output_shape, fill_value, dtype=tiles[0].dtype if tiles else bool)
    for tile, (row, col) in zip(tiles, coords):
        out[row:row + tile.shape[-2], col:col + tile.shape[-1]] = tile
    return out
