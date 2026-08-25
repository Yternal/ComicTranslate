from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DEFAULT_SERVER_URL = "http://127.0.0.1:8080/v1"
DEFAULT_DETECTOR_MODEL = Path("/Volumes/yrq/models/comic-text-and-bubble-detector")
DEFAULT_QWEN_MODEL = Path(
    "/Volumes/yrq/models/Qwen/Qwen3.5-9B-Official-MLX-4bit"
)
DEFAULT_TEXT_MASK_MODEL = Path(
    "/Volumes/yrq/models/manga-image-translator/comictextdetector.pt.onnx"
)
DEFAULT_LAMA_MODEL = Path("/Volumes/yrq/models/lama/big-lama.pt")


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    server_url: str = DEFAULT_SERVER_URL
    font_path: Path | None = None
    debug_dir: Path | None = None
    detection_threshold: float = 0.3
    nms_iou_threshold: float = 0.5
    translation_batch_size: int = 1
    server_start_timeout: float = 180.0
    minimum_font_size: int = 10
    maximum_font_size: int = 160
    text_placement: Literal["bubble", "original"] = "bubble"
    detector_model: Path = DEFAULT_DETECTOR_MODEL
    qwen_model: Path = DEFAULT_QWEN_MODEL
    text_mask_model: Path = DEFAULT_TEXT_MASK_MODEL
    lama_model: Path = DEFAULT_LAMA_MODEL

    def __post_init__(self) -> None:
        for field_name in (
            "detector_model",
            "qwen_model",
            "text_mask_model",
            "lama_model",
        ):
            value = getattr(self, field_name)
            path = Path(value).expanduser()
            if not path.is_absolute():
                raise ValueError(f"{field_name} 必须是绝对路径")
            object.__setattr__(self, field_name, path)
        if self.font_path is not None:
            object.__setattr__(self, "font_path", Path(self.font_path))
        if self.debug_dir is not None:
            object.__setattr__(self, "debug_dir", Path(self.debug_dir))
        if not 0 < self.detection_threshold <= 1:
            raise ValueError("detection_threshold 必须在 (0, 1] 范围内")
        if not 0 < self.nms_iou_threshold <= 1:
            raise ValueError("nms_iou_threshold 必须在 (0, 1] 范围内")
        if not 1 <= self.translation_batch_size <= 16:
            raise ValueError("translation_batch_size 必须在 1 到 16 之间")
        if self.minimum_font_size < 1 or self.maximum_font_size < self.minimum_font_size:
            raise ValueError("字体大小范围无效")
        if self.text_placement not in {"bubble", "original"}:
            raise ValueError("text_placement 必须是 bubble 或 original")
