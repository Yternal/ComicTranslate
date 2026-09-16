from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from comictranslate.detection import RTDetrDetector
from comictranslate.geometry import build_regions
from comictranslate.inpainting import LamaInpainter
from comictranslate.masking import ComicTextMasker


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_INTEGRATION") != "1",
        reason="set RUN_MODEL_INTEGRATION=1 to run local model inference",
    ),
]

REFERENCE = os.environ.get("COMICTRANSLATE_REFERENCE_IMAGE")
DEVICE = os.environ.get("COMICTRANSLATE_DEVICE", "cpu")
DETECTOR_MODEL = os.environ.get("COMICTRANSLATE_DETECTOR_MODEL")
TEXT_MASK_MODEL = os.environ.get("COMICTRANSLATE_TEXT_MASK_MODEL")
LAMA_MODEL = os.environ.get("COMICTRANSLATE_LAMA_MODEL")


def _model_path(value: str | None, variable: str) -> Path:
    if value is None:
        pytest.skip(f"set {variable} to run this model integration test")
    return Path(value)


def test_real_rtdetr_detects_reference_regions() -> None:
    model_path = _model_path(
        DETECTOR_MODEL, "COMICTRANSLATE_DETECTOR_MODEL"
    )
    image = Image.open(_model_path(REFERENCE, "COMICTRANSLATE_REFERENCE_IMAGE")).convert("RGB")
    detector = RTDetrDetector(model_path, 0.3, device=DEVICE)
    raw = detector.detect(image)
    normalized, regions = build_regions(raw, *image.size)
    assert {item.label for item in normalized} == {"bubble", "text_bubble", "text_free"}
    assert regions
    assert [region.id for region in regions] == [
        f"region-{index:04d}" for index in range(1, len(regions) + 1)
    ]


def test_real_onnx_refined_mask_is_binary_and_nonempty() -> None:
    model_path = _model_path(
        TEXT_MASK_MODEL, "COMICTRANSLATE_TEXT_MASK_MODEL"
    )
    image = Image.open(_model_path(REFERENCE, "COMICTRANSLATE_REFERENCE_IMAGE")).convert("RGB")
    roi = image.crop((68, 128, 225, 280))
    masker = ComicTextMasker(model_path)
    mask = masker.mask(roi)
    assert mask.shape == (roi.height, roi.width)
    assert set(np.unique(mask)) <= {0, 255}
    assert int((mask > 0).sum()) > 100


def test_real_lama_pads_and_crops_back_to_original_size() -> None:
    model_path = _model_path(LAMA_MODEL, "COMICTRANSLATE_LAMA_MODEL")
    pixels = np.full((67, 65, 3), 255, np.uint8)
    pixels[25:42, 25:42] = 0
    mask = np.zeros((67, 65), np.uint8)
    mask[25:42, 25:42] = 255
    image = Image.fromarray(pixels)
    result = LamaInpainter(model_path, device=DEVICE).inpaint(image, mask)
    assert result.size == image.size
    assert np.any(np.asarray(result) != pixels)


def test_real_qwen_and_full_pipeline(tmp_path: Path) -> None:
    from comictranslate.config import PipelineConfig
    from comictranslate.pipeline import translate_directory, translate_image
    from comictranslate.validation import resolve_device

    model = _model_path(os.environ.get("COMICTRANSLATE_QWEN_MODEL"), "COMICTRANSLATE_QWEN_MODEL")
    reference = _model_path(REFERENCE, "COMICTRANSLATE_REFERENCE_IMAGE")
    config = PipelineConfig(
        detector_model=_model_path(DETECTOR_MODEL, "COMICTRANSLATE_DETECTOR_MODEL"),
        text_mask_model=_model_path(TEXT_MASK_MODEL, "COMICTRANSLATE_TEXT_MASK_MODEL"),
        lama_model=_model_path(LAMA_MODEL, "COMICTRANSLATE_LAMA_MODEL"),
        qwen_model=model,
        qwen_service_mode=os.environ.get("COMICTRANSLATE_QWEN_MODE", "auto"),
        qwen_server_executable=Path(os.environ["COMICTRANSLATE_QWEN_SERVER"]) if os.environ.get("COMICTRANSLATE_QWEN_SERVER") else None,
        qwen_mmproj=Path(os.environ["COMICTRANSLATE_QWEN_MMPROJ"]) if os.environ.get("COMICTRANSLATE_QWEN_MMPROJ") else None,
        device=DEVICE,
        debug_dir=tmp_path / "调试 目录",
    )
    resolve_device(DEVICE)
    output = tmp_path / "单图.png"
    result = translate_image(reference, output, config)
    assert result.translated_regions + result.skipped_regions > 0
    with Image.open(reference) as source, Image.open(output) as translated:
        assert translated.size == source.size
        if "A" in source.getbands():
            assert translated.getchannel("A").tobytes() == source.getchannel("A").tobytes()
    inputs = tmp_path / "中文 输入"
    inputs.mkdir()
    (inputs / reference.name).write_bytes(reference.read_bytes())
    batch = translate_directory(inputs, tmp_path / "批量 输出", config)
    assert batch.succeeded == 1 and batch.failed == 0
