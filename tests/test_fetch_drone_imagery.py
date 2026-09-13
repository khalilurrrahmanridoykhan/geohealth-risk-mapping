import pytest

from scripts.fetch_drone_imagery import pick_scene_by_title_substring, real_asset_url


def test_pick_scene_by_title_substring_finds_a_case_insensitive_match():
    results = [
        {"title": "M2_19Feb22_Digun_Khal_1"},
        {"title": "M1_19Feb22_Rupnagar_Khal_3_transparent_mosaic_group1_22"},
    ]
    scene = pick_scene_by_title_substring(results, "rupnagar_khal_3")
    assert scene["title"].startswith("M1_19Feb22_Rupnagar_Khal_3")


def test_pick_scene_by_title_substring_raises_when_nothing_matches():
    results = [{"title": "Some Other Scene"}]
    with pytest.raises(ValueError):
        pick_scene_by_title_substring(results, "not_present")


def test_pick_scene_by_title_substring_handles_a_missing_title_key():
    results = [{"no_title_here": True}, {"title": "Target Scene"}]
    scene = pick_scene_by_title_substring(results, "target")
    assert scene["title"] == "Target Scene"


def test_real_asset_url_swaps_the_thumbnail_png_for_a_tif_on_the_same_bucket():
    scene = {
        "properties": {
            "thumbnail": "https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/abc/0/def.png",
        }
    }
    assert real_asset_url(scene) == "https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/abc/0/def.tif"


def test_real_asset_url_rejects_an_unexpected_thumbnail_format():
    scene = {"properties": {"thumbnail": "https://example.com/not-a-png.jpg"}}
    with pytest.raises(ValueError):
        real_asset_url(scene)
