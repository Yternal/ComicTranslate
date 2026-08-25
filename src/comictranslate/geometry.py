from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from .models import BBox, Detection, Region


def intersection_area(a: BBox, b: BBox) -> float:
    width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return width * height


def iou(a: BBox, b: BBox) -> float:
    intersection = intersection_area(a, b)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def containment_ratio(container: BBox, contained: BBox) -> float:
    if contained.area <= 0:
        return 0.0
    return intersection_area(container, contained) / contained.area


def union_bbox(boxes: Iterable[BBox]) -> BBox:
    values = list(boxes)
    if not values:
        raise ValueError("至少需要一个 bbox")
    return BBox(
        min(box.x1 for box in values),
        min(box.y1 for box in values),
        max(box.x2 for box in values),
        max(box.y2 for box in values),
    )


def clip_detections(
    detections: Iterable[Detection], width: int, height: int
) -> list[Detection]:
    clipped: list[Detection] = []
    for detection in detections:
        bbox = detection.bbox.clamp(width, height)
        if bbox.area <= 0:
            continue
        clipped.append(Detection(detection.label, detection.score, bbox))
    return clipped


def nms_same_class(
    detections: Sequence[Detection], iou_threshold: float = 0.5
) -> list[Detection]:
    by_label: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_label[detection.label].append(detection)

    kept: list[Detection] = []
    for label in sorted(by_label):
        remaining = sorted(
            by_label[label],
            key=lambda item: (
                -item.score,
                item.bbox.y1,
                -item.bbox.x1,
                item.bbox.y2,
                item.bbox.x2,
            ),
        )
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [
                item
                for item in remaining
                if iou(best.bbox, item.bbox) <= iou_threshold
                and not (
                    item.bbox.area > best.bbox.area
                    and containment_ratio(item.bbox, best.bbox) >= 0.9
                )
            ]
    return kept


def _overlap_ratio(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    denominator = min(end_a - start_a, end_b - start_b)
    return overlap / denominator if denominator > 0 else 0.0


def _axis_gap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if end_a < start_b:
        return start_b - end_a
    if end_b < start_a:
        return start_a - end_b
    return 0.0


def should_merge_free_text(a: BBox, b: BBox) -> bool:
    horizontal_overlap = _overlap_ratio(a.x1, a.x2, b.x1, b.x2)
    vertical_gap = _axis_gap(a.y1, a.y2, b.y1, b.y2)
    vertically_adjacent = (
        horizontal_overlap >= 0.6
        and vertical_gap <= min(a.height, b.height) * 0.25
    )

    vertical_overlap = _overlap_ratio(a.y1, a.y2, b.y1, b.y2)
    horizontal_gap = _axis_gap(a.x1, a.x2, b.x1, b.x2)
    horizontally_adjacent = (
        vertical_overlap >= 0.6
        and horizontal_gap <= min(a.width, b.width) * 0.25
    )
    return vertically_adjacent or horizontally_adjacent


def merge_free_text_boxes(boxes: Sequence[BBox]) -> list[BBox]:
    if not boxes:
        return []

    parents = list(range(len(boxes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(boxes)):
        for right in range(left + 1, len(boxes)):
            if should_merge_free_text(boxes[left], boxes[right]):
                join(left, right)

    groups: dict[int, list[BBox]] = defaultdict(list)
    for index, box in enumerate(boxes):
        groups[find(index)].append(box)
    return [union_bbox(group) for group in groups.values()]


def _matching_bubble(text: BBox, bubbles: Sequence[BBox]) -> BBox | None:
    center_x, center_y = text.center
    matches: list[BBox] = []
    for bubble in bubbles:
        contains_center = (
            bubble.x1 <= center_x <= bubble.x2 and bubble.y1 <= center_y <= bubble.y2
        )
        coverage = intersection_area(text, bubble) / text.area if text.area else 0.0
        if contains_center and coverage >= 0.8:
            matches.append(bubble)
    return min(matches, key=lambda item: item.area) if matches else None


def _crop_bbox(text_bbox: BBox, width: int, height: int) -> BBox:
    padding = max(8.0, min(text_bbox.width, text_bbox.height) * 0.05)
    return BBox(
        text_bbox.x1 - padding,
        text_bbox.y1 - padding,
        text_bbox.x2 + padding,
        text_bbox.y2 + padding,
    ).clamp(width, height)


def build_regions(
    detections: Sequence[Detection],
    width: int,
    height: int,
    *,
    nms_iou_threshold: float = 0.5,
) -> tuple[list[Detection], list[Region]]:
    normalized = nms_same_class(
        clip_detections(detections, width, height), nms_iou_threshold
    )
    bubbles = [item.bbox for item in normalized if item.label == "bubble"]
    bubble_groups: dict[BBox, list[BBox]] = defaultdict(list)
    free_boxes = [item.bbox for item in normalized if item.label == "text_free"]

    for detection in (item for item in normalized if item.label == "text_bubble"):
        bubble = _matching_bubble(detection.bbox, bubbles)
        if bubble is None:
            free_boxes.append(detection.bbox)
        else:
            bubble_groups[bubble].append(detection.bbox)

    drafts: list[tuple[str, BBox, BBox | None]] = []
    for bubble, text_boxes in bubble_groups.items():
        drafts.append(("bubble", union_bbox(text_boxes), bubble))
    for text_bbox in merge_free_text_boxes(free_boxes):
        drafts.append(("free", text_bbox, None))

    drafts.sort(
        key=lambda item: (
            round(item[1].y1, 3),
            -round(item[1].x1, 3),
            round(item[1].y2, 3),
            -round(item[1].x2, 3),
            item[0],
        )
    )
    regions = [
        Region(
            id=f"region-{index:04d}",
            kind=kind,  # type: ignore[arg-type]
            text_bbox=text_bbox,
            crop_bbox=_crop_bbox(text_bbox, width, height),
            bubble_bbox=bubble,
        )
        for index, (kind, text_bbox, bubble) in enumerate(drafts, start=1)
    ]
    return normalized, regions
