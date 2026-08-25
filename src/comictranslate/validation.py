from __future__ import annotations

import platform
import sys
from pathlib import Path

from .config import PipelineConfig
from .errors import ConfigurationError
from .rendering import resolve_font_path


def ensure_apple_metal() -> None:
    if sys.version_info[:3] != (3, 10, 18):
        actual = ".".join(str(item) for item in sys.version_info[:3])
        raise ConfigurationError(f"必须使用 Python 3.10.18，当前为 {actual}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ConfigurationError("comictranslate v1 仅支持 Apple Silicon macOS")
    try:
        import mlx.core as mx

        if not mx.metal.is_available():
            raise RuntimeError("Metal device unavailable")
    except Exception as exc:
        raise ConfigurationError(f"Apple Metal 环境不可用: {exc}") from exc


def validate_models_and_font(config: PipelineConfig) -> Path:
    required = (
        ("RT-DETR", config.detector_model, True),
        ("Qwen", config.qwen_model, True),
        ("文字 mask ONNX", config.text_mask_model, False),
        ("LaMa", config.lama_model, False),
    )
    missing = [
        f"{name}: {path}"
        for name, path, is_directory in required
        if not (path.is_dir() if is_directory else path.is_file())
    ]
    if missing:
        raise ConfigurationError("缺少本地模型：" + "；".join(missing))
    return resolve_font_path(config.font_path)
