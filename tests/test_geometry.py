from comictranslate.geometry import (
    build_regions,
    merge_free_text_boxes,
    nms_same_class,
    should_merge_free_text,
)
from comictranslate.models import BBox, Detection


def test_bbox_clipping_and_same_class_nms() -> None:
    detections = [
        Detection("text_free", 0.9, BBox(-5, -5, 20, 20)),
        Detection("text_free", 0.8, BBox(0, 0, 19, 19)),
        Detection("bubble", 0.7, BBox(0, 0, 19, 19)),
    ]
    normalized, _regions = build_regions(detections, 100, 100, nms_iou_threshold=0.5)
    assert len([item for item in normalized if item.label == "text_free"]) == 1
    assert len([item for item in normalized if item.label == "bubble"]) == 1
    assert normalized[1].bbox.area > 0


def test_nms_keeps_non_overlapping_detections() -> None:
    detections = [
        Detection("text_free", 0.9, BBox(0, 0, 10, 10)),
        Detection("text_free", 0.8, BBox(20, 20, 30, 30)),
    ]
    assert len(nms_same_class(detections)) == 2


def test_nms_removes_lower_scored_enclosing_duplicate() -> None:
    upper = Detection("text_bubble", 0.9, BBox(10, 0, 90, 90))
    aggregate = Detection("text_bubble", 0.8, BBox(8, 0, 92, 190))
    lower = Detection("text_bubble", 0.7, BBox(10, 100, 90, 190))

    kept = nms_same_class([aggregate, lower, upper])

    assert kept == [upper, lower]


def test_build_regions_drops_aggregate_bubble_and_text_detections() -> None:
    detections = [
        Detection("bubble", 0.9, BBox(0, 0, 100, 100)),
        Detection("bubble", 0.85, BBox(0, 100, 100, 200)),
        Detection("bubble", 0.5, BBox(0, 0, 100, 200)),
        Detection("text_bubble", 0.9, BBox(10, 10, 90, 90)),
        Detection("text_bubble", 0.8, BBox(10, 10, 90, 190)),
        Detection("text_bubble", 0.7, BBox(10, 110, 90, 190)),
    ]

    _normalized, regions = build_regions(detections, 100, 200)

    assert [region.text_bbox for region in regions] == [
        BBox(10, 10, 90, 90),
        BBox(10, 110, 90, 190),
    ]


def test_bubble_association_uses_smallest_matching_bubble_and_merges_text() -> None:
    small = BBox(10, 10, 55, 70)
    detections = [
        Detection("bubble", 0.9, BBox(0, 0, 100, 100)),
        Detection("bubble", 0.8, small),
        Detection("text_bubble", 0.9, BBox(20, 20, 40, 35)),
        Detection("text_bubble", 0.8, BBox(20, 40, 40, 55)),
    ]
    _normalized, regions = build_regions(detections, 120, 120)
    assert len(regions) == 1
    assert regions[0].kind == "bubble"
    assert regions[0].bubble_bbox == small
    assert regions[0].text_bbox == BBox(20, 20, 40, 55)
    assert regions[0].id == "region-0001"


def test_bubble_requires_center_and_eighty_percent_coverage() -> None:
    detections = [
        Detection("bubble", 0.9, BBox(0, 0, 50, 50)),
        Detection("text_bubble", 0.9, BBox(40, 40, 80, 80)),
    ]
    _normalized, regions = build_regions(detections, 100, 100)
    assert regions[0].kind == "free"


def test_free_text_merge_respects_overlap_and_gap_thresholds() -> None:
    first = BBox(0, 0, 20, 20)
    close_vertical = BBox(2, 24, 22, 44)
    too_far = BBox(2, 26, 22, 46)
    assert should_merge_free_text(first, close_vertical)
    assert not should_merge_free_text(first, too_far)
    assert merge_free_text_boxes([first, close_vertical]) == [BBox(0, 0, 22, 44)]


def test_region_crop_uses_eight_pixel_minimum_padding_and_clips() -> None:
    detections = [Detection("text_free", 0.9, BBox(2, 3, 12, 13))]
    _normalized, regions = build_regions(detections, 20, 20)
    assert regions[0].crop_bbox == BBox(0, 0, 20, 20)
