from __future__ import annotations

from pathlib import Path

import pytest

from comictranslate.config import PipelineConfig
from comictranslate.rendering import resolve_font_path


@pytest.fixture
def local_config(tmp_path: Path) -> PipelineConfig:
    models = tmp_path / "models"
    (models / "comic-text-and-bubble-detector").mkdir(parents=True)
    (models / "Qwen" / "Qwen3.5-9B-Official-MLX-4bit").mkdir(parents=True)
    mask_path = models / "manga-image-translator" / "comictextdetector.pt.onnx"
    lama_path = models / "lama" / "big-lama.pt"
    mask_path.parent.mkdir(parents=True)
    lama_path.parent.mkdir(parents=True)
    mask_path.touch()
    lama_path.touch()
    return PipelineConfig(
        detector_model=models / "comic-text-and-bubble-detector",
        qwen_model=models / "Qwen" / "Qwen3.5-9B-Official-MLX-4bit",
        text_mask_model=mask_path,
        lama_model=lama_path,
        font_path=resolve_font_path(),
    )
