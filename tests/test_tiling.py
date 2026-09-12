import numpy as np
import pytest

from src.tiling import assign_tile_split, tile_array


def test_tile_array_splits_an_exact_grid():
    arr = np.zeros((512, 512))
    tiles, coords = tile_array(arr, tile_size=256)
    assert len(tiles) == 4
    assert all(t.shape == (256, 256) for t in tiles)
    assert set(coords) == {(0, 0), (0, 256), (256, 0), (256, 256)}


def test_tile_array_drops_incomplete_edge_tiles():
    arr = np.zeros((600, 300))  # 600/256 = 2 full rows + a 88px remainder; 300/256 = 1 full col + 44px remainder
    tiles, coords = tile_array(arr, tile_size=256)
    assert len(tiles) == 2  # only the 2 full 256x256 tiles in the single full column
    assert all(t.shape == (256, 256) for t in tiles)


def test_tile_array_preserves_leading_channel_dimension():
    arr = np.zeros((7, 256, 300))  # (channels, H, W)
    tiles, coords = tile_array(arr, tile_size=256)
    assert len(tiles) == 1
    assert tiles[0].shape == (7, 256, 256)


def test_tile_array_rejects_a_non_positive_tile_size():
    with pytest.raises(ValueError):
        tile_array(np.zeros((10, 10)), tile_size=0)


def test_assign_tile_split_holds_out_the_eastern_fraction():
    coords = [(0, 0), (0, 256), (0, 512), (0, 768)]  # 4 tiles across a width of 1024
    result = assign_tile_split(coords, width=1024, test_fraction=0.3)
    # split_col = 1024 * 0.7 = 716.8 -- only the tile starting at col 768 clears it
    assert result == [False, False, False, True]


def test_assign_tile_split_rejects_an_invalid_fraction():
    with pytest.raises(ValueError):
        assign_tile_split([(0, 0)], width=100, test_fraction=0)
