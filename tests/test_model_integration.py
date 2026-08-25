from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from comictranslate.config import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_LAMA_MODEL,
    DEFAULT_TEXT_MASK_MODEL,
)
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

REFERENCE = Path(
    os.environ.get(
        "COMICTRANSLATE_REFERENCE_IMAGE",
        "/Users/yaoruiqi/program/python/llmTest/1.jpeg",
    )
)


def test_real_rtdetr_detects_reference_regions() -> None:
    image = Image.open(REFERENCE).convert("RGB")
    detector = RTDetrDetector(DEFAULT_DETECTOR_MODEL, 0.3)
    raw = detector.detect(image)
    normalized, regions = build_regions(raw, *image.size)
    assert {item.label for item in normalized} == {"bubble", "text_bubble", "text_free"}
    assert regions
    assert [region.id for region in regions] == [
        f"region-{index:04d}" for index in range(1, len(regions) + 1)
    ]


def test_real_onnx_refined_mask_is_binary_and_nonempty() -> None:
    image = Image.open(REFERENCE).convert("RGB")
    roi = image.crop((68, 128, 225, 280))
    masker = ComicTextMasker(DEFAULT_TEXT_MASK_MODEL)
    mask = masker.mask(roi)
    assert mask.shape == (roi.height, roi.width)
    assert set(np.unique(mask)) <= {0, 255}
    assert int((mask > 0).sum()) > 100


def test_real_lama_pads_and_crops_back_to_original_size() -> None:
    pixels = np.full((67, 65, 3), 255, np.uint8)
    pixels[25:42, 25:42] = 0
    mask = np.zeros((67, 65), np.uint8)
    mask[25:42, 25:42] = 255
    image = Image.fromarray(pixels)
    result = LamaInpainter(DEFAULT_LAMA_MODEL).inpaint(image, mask)
    assert result.size == image.size
    assert np.any(np.asarray(result) != pixels)
