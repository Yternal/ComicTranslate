from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def clamp(self, width: int, height: int) -> BBox:
        return BBox(
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            min(max(self.x2, 0.0), float(width)),
            min(max(self.y2, 0.0), float(height)),
        )

    def as_int(self) -> tuple[int, int, int, int]:
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))

    def to_list(self) -> list[float]:
        return [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)]


@dataclass(frozen=True, slots=True)
class Detection:
    label: Literal["bubble", "text_bubble", "text_free"]
    score: float
    bbox: BBox

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "score": round(self.score, 4), "bbox": self.bbox.to_list()}


@dataclass(frozen=True, slots=True)
class Region:
    id: str
    kind: Literal["bubble", "free"]
    text_bbox: BBox
    crop_bbox: BBox
    bubble_bbox: BBox | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text_bbox": self.text_bbox.to_list(),
            "crop_bbox": self.crop_bbox.to_list(),
            "bubble_bbox": self.bubble_bbox.to_list() if self.bubble_bbox else None,
        }


@dataclass(frozen=True, slots=True)
class Translation:
    id: str
    source_text: str
    action: Literal["translate", "skip"]
    translation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "source_text": self.source_text,
            "action": self.action,
            "translation": self.translation,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    width: int
    height: int
    translated_regions: int
    skipped_regions: int
    output_path: Path
    region_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "translated_regions": self.translated_regions,
            "skipped_regions": self.skipped_regions,
            "output_path": str(self.output_path),
            "region_ids": list(self.region_ids),
        }


@dataclass(frozen=True, slots=True)
class BatchFailure:
    input_path: Path
    output_path: Path
    error_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class BatchResult:
    input_dir: Path
    output_dir: Path
    results: tuple[PipelineResult, ...] = field(default_factory=tuple)
    skipped_inputs: tuple[Path, ...] = field(default_factory=tuple)
    failures: tuple[BatchFailure, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return self.succeeded + self.skipped + self.failed

    @property
    def succeeded(self) -> int:
        return len(self.results)

    @property
    def skipped(self) -> int:
        return len(self.skipped_inputs)

    @property
    def failed(self) -> int:
        return len(self.failures)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "total": self.total,
            "succeeded": self.succeeded,
            "skipped": self.skipped,
            "failed": self.failed,
            "results": [result.to_dict() for result in self.results],
            "skipped_inputs": [str(path) for path in self.skipped_inputs],
            "failures": [failure.to_dict() for failure in self.failures],
        }
