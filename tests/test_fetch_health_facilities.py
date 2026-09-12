from scripts.fetch_health_facilities import build_overpass_query, overpass_elements_to_geojson


def test_build_overpass_query_reorders_bbox_to_overpass_south_west_north_east():
    query = build_overpass_query((91.15, 24.5, 91.5, 24.8), ["hospital"])
    # Overpass wants (south, west, north, east); our bbox is (min_lon, min_lat, max_lon, max_lat)
    assert "24.5,91.15,24.8,91.5" in query


def test_build_overpass_query_joins_multiple_amenities_with_pipe():
    query = build_overpass_query((0, 0, 1, 1), ["hospital", "clinic"])
    assert "hospital|clinic" in query


def test_overpass_elements_to_geojson_handles_a_node():
    elements = [{"type": "node", "id": 1, "lat": 24.6, "lon": 91.3, "tags": {"amenity": "clinic", "name": "Test Clinic"}}]
    geojson = overpass_elements_to_geojson(elements)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 1
    feature = geojson["features"][0]
    assert feature["geometry"]["coordinates"] == [91.3, 24.6]
    assert feature["properties"]["name"] == "Test Clinic"


def test_overpass_elements_to_geojson_handles_a_way_using_its_center():
    elements = [{"type": "way", "id": 2, "center": {"lat": 24.7, "lon": 91.4}, "tags": {"amenity": "hospital"}}]
    geojson = overpass_elements_to_geojson(elements)
    assert geojson["features"][0]["geometry"]["coordinates"] == [91.4, 24.7]


def test_overpass_elements_to_geojson_skips_elements_with_no_location():
    elements = [{"type": "relation", "id": 3, "tags": {"amenity": "hospital"}}]
    geojson = overpass_elements_to_geojson(elements)
    assert len(geojson["features"]) == 0


def test_overpass_elements_to_geojson_handles_missing_tags():
    elements = [{"type": "node", "id": 4, "lat": 1.0, "lon": 2.0}]
    geojson = overpass_elements_to_geojson(elements)
    assert geojson["features"][0]["properties"]["name"] is None
