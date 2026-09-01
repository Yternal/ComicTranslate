from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Literal


DEFAULT_SERVER_URL = "http://127.0.0.1:8080/v1"
DEFAULT_DETECTOR_MODEL: Path | None = None
DEFAULT_QWEN_MODEL: Path | None = None
DEFAULT_TEXT_MASK_MODEL: Path | None = None
DEFAULT_LAMA_MODEL: Path | None = None


def is_absolute_path(value: str | Path) -> bool:
    text = str(value)
    return Path(text).expanduser().is_absolute() or PureWindowsPath(text).is_absolute()


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
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    qwen_service_mode: Literal["auto", "managed-mlx", "external"] = "auto"
    qwen_model_id: str | None = None
    detector_model: Path | None = DEFAULT_DETECTOR_MODEL
    qwen_model: Path | None = DEFAULT_QWEN_MODEL
    text_mask_model: Path | None = DEFAULT_TEXT_MASK_MODEL
    lama_model: Path | None = DEFAULT_LAMA_MODEL

    def __post_init__(self) -> None:
        for field_name in (
            "detector_model",
            "qwen_model",
            "text_mask_model",
            "lama_model",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            path = Path(value).expanduser()
            if not is_absolute_path(value):
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
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device 必须是 auto、cpu、cuda 或 mps")
        if self.qwen_service_mode not in {"auto", "managed-mlx", "external"}:
            raise ValueError(
                "qwen_service_mode 必须是 auto、managed-mlx 或 external"
            )
        if self.qwen_model_id is not None:
            model_id = self.qwen_model_id.strip()
            if not model_id:
                raise ValueError("qwen_model_id 不能为空")
            object.__setattr__(self, "qwen_model_id", model_id)
