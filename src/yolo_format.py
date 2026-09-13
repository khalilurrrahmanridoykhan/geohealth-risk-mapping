"""Phase H6 -- polygon geometries to YOLO-format bounding boxes. Same core
technique already proven in the companion best-model-for-satellite-imagery
repo's src/labels.py (that repo's own real bug: naive pixel bounds without
going through the raster's affine transform produces nonsense boxes against
real georeferenced data -- avoided here from the start by requiring an
explicit transform argument, not by discovering it the hard way again).

Pure, unit-tested with synthetic geometries in tests/test_yolo_format.py.
"""

from __future__ import annotations

from affine import Affine
from shapely.geometry.base import BaseGeometry


def polygon_to_yolo_box(
    polygon: BaseGeometry, image_width: int, image_height: int, transform: Affine, class_id: int = 0,
) -> tuple[int, float, float, float, float]:
    """A polygon's bounding box, in real-world (georeferenced) coordinates,
    converted to YOLO's normalized (class, x_center, y_center, width,
    height) format in pixel space. Uses all 4 corners of the bounding box
    (not just 2), so a transform with rotation/skew is still handled
    correctly."""
    minx, miny, maxx, maxy = polygon.bounds
    inverse = ~transform
    corners = [inverse @ (x, y) for x, y in [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    px_min, px_max = min(xs), max(xs)
    py_min, py_max = min(ys), max(ys)

    x_center = (px_min + px_max) / 2 / image_width
    y_center = (py_min + py_max) / 2 / image_height
    width = (px_max - px_min) / image_width
    height = (py_max - py_min) / image_height
    return class_id, x_center, y_center, width, height


def polygons_to_yolo_boxes(
    polygons: list[BaseGeometry], image_width: int, image_height: int, transform: Affine, class_id: int = 0,
) -> list[tuple[int, float, float, float, float]]:
    return [polygon_to_yolo_box(p, image_width, image_height, transform, class_id) for p in polygons]


def format_yolo_label_file(boxes: list[tuple[int, float, float, float, float]]) -> str:
    """Ultralytics' plain-text label format: one 'class x y w h' line per box."""
    return "\n".join(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in boxes)
