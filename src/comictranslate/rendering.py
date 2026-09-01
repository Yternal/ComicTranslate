from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .errors import ConfigurationError
from .models import BBox, Region, Translation


MACOS_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)
WINDOWS_FONT_FILENAMES = ("msyh.ttc", "simhei.ttf", "simsun.ttc")

# An axis-aligned rectangle needs roughly 14.65% inset on every side to fit
# inside an ellipse. Use 18% to leave room for irregular borders, tails,
# anti-aliasing, and the contrasting text stroke.
BUBBLE_SAFE_INSET_RATIO = 0.18
BUBBLE_SAFE_INSET_MINIMUM = 4.0


def resolve_font_path(font_path: Path | None = None) -> Path:
    if font_path is not None:
        candidate = Path(font_path).expanduser()
        if candidate.is_file():
            return candidate
        raise ConfigurationError(f"字体文件不存在: {candidate}")
    if platform.system() == "Windows":
        windir = os.environ.get("WINDIR")
        candidates = (
            tuple(Path(windir, "Fonts", name) for name in WINDOWS_FONT_FILENAMES)
            if windir
            else ()
        )
    else:
        candidates = MACOS_FONT_CANDIDATES
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ConfigurationError("找不到可用的中文系统字体，请使用 --font 指定字体")


@dataclass(frozen=True, slots=True)
class TextLayout:
    orientation: Literal["horizontal", "vertical"]
    font_size: int
    lines: tuple[str, ...]
    columns: tuple[str, ...]
    width: float
    height: float


def _text_bbox(font: ImageFont.FreeTypeFont, text: str, stroke_width: int) -> tuple[int, int, int, int]:
    return font.getbbox(text or "国", stroke_width=stroke_width)


def _text_size(font: ImageFont.FreeTypeFont, text: str, stroke_width: int) -> tuple[int, int]:
    left, top, right, bottom = _text_bbox(font, text, stroke_width)
    return right - left, bottom - top


def _wrap_horizontal(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: float,
    stroke_width: int,
) -> tuple[str, ...]:
    lines: list[str] = []
    paragraphs = text.splitlines() or [text]
    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and _text_size(font, candidate, stroke_width)[0] > max_width:
                lines.append(current.rstrip())
                current = character.lstrip()
            else:
                current = candidate
        lines.append(current.rstrip())
    return tuple(lines)


def _horizontal_layout(
    text: str, font: ImageFont.FreeTypeFont, font_size: int, box: BBox
) -> TextLayout:
    stroke = max(1, round(font_size * 0.06))
    lines = _wrap_horizontal(text, font, box.width, stroke)
    sizes = [_text_size(font, line, stroke) for line in lines]
    spacing = max(1, round(font_size * 0.16))
    width = max((size[0] for size in sizes), default=0)
    height = sum(size[1] for size in sizes) + spacing * max(0, len(lines) - 1)
    return TextLayout("horizontal", font_size, lines, (), width, height)


def _vertical_layout(
    text: str, font: ImageFont.FreeTypeFont, font_size: int, box: BBox
) -> TextLayout:
    characters = "".join(text.split())
    stroke = max(1, round(font_size * 0.06))
    glyph_sizes = [_text_size(font, character, stroke) for character in characters]
    cell_width = max((size[0] for size in glyph_sizes), default=0)
    cell_height = max((size[1] for size in glyph_sizes), default=0)
    spacing = max(1, round(font_size * 0.12))
    row_step = cell_height + spacing
    rows = max(1, int((box.height + spacing) // max(1, row_step)))
    columns = tuple(
        characters[index : index + rows] for index in range(0, len(characters), rows)
    )
    width = len(columns) * cell_width + max(0, len(columns) - 1) * spacing
    height = min(rows, len(characters)) * cell_height
    if characters:
        height += max(0, min(rows, len(characters)) - 1) * spacing
    return TextLayout("vertical", font_size, (), columns, width, height)


def fit_text(
    text: str,
    box: BBox,
    font_path: Path,
    *,
    minimum_size: int = 10,
    maximum_size: int = 160,
) -> tuple[TextLayout, ImageFont.FreeTypeFont]:
    if not text.strip():
        raise ValueError("译文不能为空")
    orientation: Literal["horizontal", "vertical"] = (
        "vertical" if box.width / max(box.height, 1) < 0.8 else "horizontal"
    )
    best: tuple[TextLayout, ImageFont.FreeTypeFont] | None = None
    low, high = minimum_size, maximum_size
    while low <= high:
        size = (low + high) // 2
        font = ImageFont.truetype(str(font_path), size=size)
        layout = (
            _vertical_layout(text, font, size, box)
            if orientation == "vertical"
            else _horizontal_layout(text, font, size, box)
        )
        fits = layout.width <= box.width and layout.height <= box.height
        if fits:
            best = (layout, font)
            low = size + 1
        else:
            high = size - 1
    if best is not None:
        return best

    # Preserve the configured minimum readable size and allow a centered
    # overflow instead of aborting the entire page.
    font = ImageFont.truetype(str(font_path), size=minimum_size)
    layout = (
        _vertical_layout(text, font, minimum_size, box)
        if orientation == "vertical"
        else _horizontal_layout(text, font, minimum_size, box)
    )
    return layout, font


def _target_bbox(
    region: Region, placement: Literal["bubble", "original"] = "bubble"
) -> BBox:
    if placement == "original" or region.bubble_bbox is None:
        return region.text_bbox
    bubble = region.bubble_bbox
    inset_x = max(BUBBLE_SAFE_INSET_MINIMUM, bubble.width * BUBBLE_SAFE_INSET_RATIO)
    inset_y = max(BUBBLE_SAFE_INSET_MINIMUM, bubble.height * BUBBLE_SAFE_INSET_RATIO)
    return BBox(
        bubble.x1 + inset_x,
        bubble.y1 + inset_y,
        bubble.x2 - inset_x,
        bubble.y2 - inset_y,
    )


def _colors(image: Image.Image, box: BBox) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    crop = np.asarray(image.crop(box.as_int()).convert("RGB"), dtype=np.float32)
    if not crop.size:
        brightness = 255.0
    else:
        brightness = float(
            np.mean(crop[..., 0] * 0.2126 + crop[..., 1] * 0.7152 + crop[..., 2] * 0.0722)
        )
    return ((0, 0, 0), (255, 255, 255)) if brightness >= 128 else (
        (255, 255, 255),
        (0, 0, 0),
    )


class ChineseTextRenderer:
    def __init__(
        self,
        font_path: Path | None = None,
        *,
        minimum_size: int = 10,
        maximum_size: int = 160,
        text_placement: Literal["bubble", "original"] = "bubble",
    ) -> None:
        self.font_path = resolve_font_path(font_path)
        self.minimum_size = minimum_size
        self.maximum_size = maximum_size
        if text_placement not in {"bubble", "original"}:
            raise ValueError("text_placement 必须是 bubble 或 original")
        self.text_placement = text_placement

    def _draw_horizontal(
        self,
        draw: ImageDraw.ImageDraw,
        box: BBox,
        layout: TextLayout,
        font: ImageFont.FreeTypeFont,
        fill: tuple[int, int, int],
        stroke_fill: tuple[int, int, int],
    ) -> None:
        stroke = max(1, round(layout.font_size * 0.06))
        spacing = max(1, round(layout.font_size * 0.16))
        sizes = [_text_size(font, line, stroke) for line in layout.lines]
        cursor_y = box.y1 + (box.height - layout.height) / 2
        for line, (width, height) in zip(layout.lines, sizes):
            left, top, _right, _bottom = _text_bbox(font, line, stroke)
            x = box.x1 + (box.width - width) / 2 - left
            y = cursor_y - top
            draw.text(
                (x, y),
                line,
                font=font,
                fill=fill,
                stroke_width=stroke,
                stroke_fill=stroke_fill,
            )
            cursor_y += height + spacing

    def _draw_vertical(
        self,
        draw: ImageDraw.ImageDraw,
        box: BBox,
        layout: TextLayout,
        font: ImageFont.FreeTypeFont,
        fill: tuple[int, int, int],
        stroke_fill: tuple[int, int, int],
    ) -> None:
        stroke = max(1, round(layout.font_size * 0.06))
        spacing = max(1, round(layout.font_size * 0.12))
        glyph_sizes = [
            _text_size(font, character, stroke)
            for column in layout.columns
            for character in column
        ]
        cell_width = max((size[0] for size in glyph_sizes), default=0)
        cell_height = max((size[1] for size in glyph_sizes), default=0)
        start_x = box.x1 + (box.width - layout.width) / 2
        for column_index, column in enumerate(layout.columns):
            used_height = len(column) * cell_height + max(0, len(column) - 1) * spacing
            cursor_y = box.y1 + (box.height - used_height) / 2
            center_x = start_x + cell_width / 2 + column_index * (cell_width + spacing)
            for character in column:
                width, _height = _text_size(font, character, stroke)
                left, top, _right, _bottom = _text_bbox(font, character, stroke)
                draw.text(
                    (center_x - width / 2 - left, cursor_y - top),
                    character,
                    font=font,
                    fill=fill,
                    stroke_width=stroke,
                    stroke_fill=stroke_fill,
                )
                cursor_y += cell_height + spacing

    def render(
        self,
        image: Image.Image,
        regions: list[Region],
        translations: list[Translation],
    ) -> Image.Image:
        result = image.convert("RGB").copy()
        draw = ImageDraw.Draw(result)
        translations_by_id = {item.id: item for item in translations}
        for region in regions:
            translation = translations_by_id.get(region.id)
            if translation is None or translation.action != "translate":
                continue
            box = _target_bbox(region, self.text_placement).clamp(*result.size)
            layout, font = fit_text(
                translation.translation,
                box,
                self.font_path,
                minimum_size=self.minimum_size,
                maximum_size=self.maximum_size,
            )
            fill, stroke_fill = _colors(result, box)
            if layout.orientation == "vertical":
                self._draw_vertical(draw, box, layout, font, fill, stroke_fill)
            else:
                self._draw_horizontal(draw, box, layout, font, fill, stroke_fill)
        return result
