from affine import Affine
from shapely.geometry import MultiPolygon, Polygon

from src.yolo_format import format_yolo_label_file, polygon_to_yolo_box, polygons_to_yolo_boxes

IDENTITY = Affine.identity()


def test_polygon_to_yolo_box_centers_and_normalizes_with_identity_transform():
    # A 10x10 pixel square centered at (50, 50) in a 100x100 image.
    square = Polygon([(45, 45), (55, 45), (55, 55), (45, 55)])
    cls, x, y, w, h = polygon_to_yolo_box(square, 100, 100, IDENTITY)
    assert cls == 0
    assert abs(x - 0.5) < 1e-9
    assert abs(y - 0.5) < 1e-9
    assert abs(w - 0.1) < 1e-9
    assert abs(h - 0.1) < 1e-9


def test_polygon_to_yolo_box_respects_a_custom_class_id():
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    cls, *_ = polygon_to_yolo_box(square, 100, 100, IDENTITY, class_id=3)
    assert cls == 3


def test_polygon_to_yolo_box_converts_through_a_real_georeferenced_transform():
    # A real-world building footprint 10m wide, in a raster at 0.1m/pixel,
    # origin at (1000, 2000), north-up.
    transform = Affine(0.1, 0.0, 1000.0, 0.0, -0.1, 2000.0)
    building = Polygon([(1010, 1990), (1011, 1990), (1011, 1991), (1010, 1991)])  # 1m x 1m at world coords
    cls, x, y, w, h = polygon_to_yolo_box(building, 20000, 20000, transform)
    # world (1010, 1990) is 10m right, 10m down from origin -> pixel (100, 100)
    # a 1m x 1m box -> 10 x 10 pixels
    assert abs(x - (100 + 5) / 20000) < 1e-6
    assert abs(w - 10 / 20000) < 1e-6


def test_polygon_to_yolo_box_handles_a_multipolygon_via_its_overall_bounds():
    multi = MultiPolygon([
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
    ])
    cls, x, y, w, h = polygon_to_yolo_box(multi, 100, 100, IDENTITY)
    assert abs(w - 0.3) < 1e-9  # bounds span 0 to 30


def test_polygons_to_yolo_boxes_returns_one_box_per_polygon_in_order():
    polygons = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(50, 50), (60, 50), (60, 60), (50, 60)]),
    ]
    boxes = polygons_to_yolo_boxes(polygons, 100, 100, IDENTITY)
    assert len(boxes) == 2
    assert boxes[0][1] < boxes[1][1]  # first polygon's x_center is smaller


def test_polygons_to_yolo_boxes_handles_an_empty_list():
    assert polygons_to_yolo_boxes([], 100, 100, IDENTITY) == []


def test_format_yolo_label_file_matches_ultralytics_format():
    boxes = [(0, 0.5, 0.5, 0.1, 0.2), (1, 0.25, 0.75, 0.05, 0.05)]
    result = format_yolo_label_file(boxes)
    lines = result.splitlines()
    assert lines[0] == "0 0.500000 0.500000 0.100000 0.200000"
    assert lines[1] == "1 0.250000 0.750000 0.050000 0.050000"


def test_format_yolo_label_file_handles_no_boxes():
    assert format_yolo_label_file([]) == ""
