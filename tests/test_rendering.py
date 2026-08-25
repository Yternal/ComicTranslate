import numpy as np
from PIL import Image, ImageDraw, ImageFont

from comictranslate.models import BBox, Region, Translation
from comictranslate.rendering import (
    ChineseTextRenderer,
    _target_bbox,
    _vertical_layout,
    fit_text,
    resolve_font_path,
)


def test_horizontal_and_vertical_layout_selection() -> None:
    font = resolve_font_path()
    horizontal, _ = fit_text("自然的中文台词", BBox(0, 0, 160, 60), font)
    vertical, _ = fit_text("竖排文字", BBox(0, 0, 40, 120), font)
    assert horizontal.orientation == "horizontal"
    assert vertical.orientation == "vertical"
    assert vertical.columns


def test_too_long_translation_uses_minimum_size_and_allows_overflow() -> None:
    font = resolve_font_path()
    box = BBox(0, 0, 3, 3)
    layout, _font = fit_text("绝对放不下", box, font, minimum_size=10)
    assert layout.font_size == 10
    assert layout.width > box.width or layout.height > box.height


def test_renderer_changes_pixels_without_changing_size() -> None:
    image = Image.new("RGB", (180, 100), "white")
    region = Region(
        "a",
        "bubble",
        BBox(40, 30, 140, 70),
        BBox(30, 20, 150, 80),
        BBox(20, 10, 160, 90),
    )
    renderer = ChineseTextRenderer(resolve_font_path())
    result = renderer.render(
        image,
        [region],
        [Translation("a", "hello", "translate", "你好")],
    )
    assert result.size == image.size
    assert result.tobytes() != image.tobytes()


def test_vertical_columns_are_drawn_from_left_to_right() -> None:
    class RecordingDraw:
        def __init__(self) -> None:
            self.calls: list[tuple[float, float, str]] = []

        def text(self, position, text, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls.append((position[0], position[1], text))

    font_path = resolve_font_path()
    font = ImageFont.truetype(str(font_path), size=20)
    box = BBox(0, 0, 100, 45)
    layout = _vertical_layout("甲乙丙丁", font, 20, box)
    assert len(layout.columns) > 1
    draw = RecordingDraw()
    renderer = ChineseTextRenderer(font_path)
    renderer._draw_vertical(draw, box, layout, font, (0, 0, 0), (255, 255, 255))  # type: ignore[arg-type]
    first_character_x = {
        character: x for x, _y, character in draw.calls if character in {column[0] for column in layout.columns}
    }
    assert [first_character_x[column[0]] for column in layout.columns] == sorted(
        first_character_x[column[0]] for column in layout.columns
    )


def test_bubble_text_pixels_stay_inside_elliptical_bubble() -> None:
    image = Image.new("RGB", (220, 140), (180, 180, 180))
    bubble_bbox = BBox(10, 10, 210, 130)
    draw = ImageDraw.Draw(image)
    draw.ellipse(bubble_bbox.as_int(), fill="white", outline="black", width=2)
    before = np.asarray(image).copy()

    region = Region(
        "a",
        "bubble",
        BBox(45, 45, 175, 95),
        BBox(35, 35, 185, 105),
        bubble_bbox,
    )
    output = ChineseTextRenderer(resolve_font_path()).render(
        image,
        [region],
        [Translation("a", "source", "translate", "这是一段需要安全排版的中文译文")],
    )
    changed = np.any(np.asarray(output) != before, axis=2)

    ellipse_mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(ellipse_mask).ellipse(bubble_bbox.as_int(), fill=255)
    outside_bubble = np.asarray(ellipse_mask) == 0
    assert not np.any(changed & outside_bubble)


def test_original_placement_uses_detected_text_bbox() -> None:
    region = Region(
        "a",
        "bubble",
        BBox(80, 40, 120, 80),
        BBox(70, 30, 130, 90),
        BBox(10, 10, 190, 110),
    )
    assert _target_bbox(region, "original") == region.text_bbox
    assert _target_bbox(region, "bubble") != region.text_bbox
