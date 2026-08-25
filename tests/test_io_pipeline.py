from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image, ImageDraw

from comictranslate.errors import ConfigurationError, TextLayoutError
from comictranslate.io_utils import default_output_path
from comictranslate.models import BBox, Detection, Translation
from comictranslate.pipeline import Pipeline


class FakeDetector:
    def detect(self, image):  # type: ignore[no-untyped-def]
        return [Detection("text_free", 0.9, BBox(12, 10, 52, 30))]


class EmptyDetector:
    def detect(self, image):  # type: ignore[no-untyped-def]
        return []


class FakeTranslator:
    def __init__(self, action: str = "translate") -> None:
        self.action = action

    def translate(self, image, regions):  # type: ignore[no-untyped-def]
        return [
            Translation(
                region.id,
                "hello" if self.action == "translate" else "中文",
                self.action,
                "你好" if self.action == "translate" else "",
            )
            for region in regions
        ]


class FakeMasker:
    def __init__(self) -> None:
        self.calls = 0

    def mask(self, roi):  # type: ignore[no-untyped-def]
        self.calls += 1
        result = np.zeros((roi.height, roi.width), np.uint8)
        result[roi.height // 2, roi.width // 2] = 255
        return result


class FakeInpainter:
    def __init__(self) -> None:
        self.mask_nonempty = False

    def inpaint(self, image, mask):  # type: ignore[no-untyped-def]
        self.mask_nonempty = bool(mask.any())
        return image.copy()


class FakeRenderer:
    def render(self, image, regions, translations):  # type: ignore[no-untyped-def]
        result = image.copy()
        draw = ImageDraw.Draw(result)
        draw.point((0, 0), fill=(1, 2, 3))
        return result


def test_mock_pipeline_preserves_dimensions_alpha_and_writes_debug(
    tmp_path: Path, local_config
) -> None:  # type: ignore[no-untyped-def]
    alpha = Image.fromarray(np.arange(80 * 60, dtype=np.uint16).reshape(60, 80).astype(np.uint8))
    source = Image.new("RGBA", (80, 60), (240, 240, 240, 255))
    source.putalpha(alpha)
    input_path = tmp_path / "source.png"
    output_path = tmp_path / "translated.png"
    source.save(input_path)
    debug_dir = tmp_path / "debug"
    stale_roi = debug_dir / "roi" / "region-9999.png"
    stale_roi.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1), "black").save(stale_roi)
    config = type(local_config)(
        detector_model=local_config.detector_model,
        qwen_model=local_config.qwen_model,
        text_mask_model=local_config.text_mask_model,
        lama_model=local_config.lama_model,
        font_path=local_config.font_path,
        debug_dir=debug_dir,
    )
    masker = FakeMasker()
    inpainter = FakeInpainter()
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        result = Pipeline(
            config,
            detector=FakeDetector(),
            translator=FakeTranslator(),
            masker=masker,
            inpainter=inpainter,
            renderer=FakeRenderer(),
            environment_check=lambda: None,
        ).run(input_path, output_path)
    finally:
        logger.remove(sink_id)
    with Image.open(output_path) as output:
        assert output.size == source.size
        assert output.getchannel("A").tobytes() == alpha.tobytes()
    assert result.translated_regions == 1
    assert result.skipped_regions == 0
    assert masker.calls == 1
    assert inpainter.mask_nonempty
    assert (debug_dir / "detections.json").is_file()
    assert (debug_dir / "roi" / "region-0001.png").is_file()
    assert not stale_roi.exists()
    assert (debug_dir / "translations.json").is_file()
    assert (debug_dir / "mask.png").is_file()
    assert (debug_dir / "clean.png").is_file()
    translations_debug = json.loads(
        (debug_dir / "translations.json").read_text(encoding="utf-8")
    )
    assert translations_debug[0]["roi_file"] == "roi/region-0001.png"
    assert translations_debug[0]["crop_bbox"] == [4.0, 2.0, 60.0, 38.0]
    log_text = "".join(messages)
    assert "[1/8] 开始：校验输入、模型与运行环境" in log_text
    assert "[8/8] 完成：保存输出图片" in log_text
    assert f"处理完成：{output_path.resolve()}" in log_text


def test_skipped_region_is_not_masked(tmp_path: Path, local_config) -> None:  # type: ignore[no-untyped-def]
    input_path = tmp_path / "source.png"
    output_path = tmp_path / "translated.png"
    Image.new("RGB", (80, 60), "white").save(input_path)
    masker = FakeMasker()
    inpainter = FakeInpainter()
    result = Pipeline(
        local_config,
        detector=FakeDetector(),
        translator=FakeTranslator("skip"),
        masker=masker,
        inpainter=inpainter,
        renderer=FakeRenderer(),
        environment_check=lambda: None,
    ).run(input_path, output_path)
    assert result.skipped_regions == 1
    assert masker.calls == 0
    assert not inpainter.mask_nonempty


def test_no_text_page_writes_unchanged_sized_output_without_translation(
    tmp_path: Path, local_config
) -> None:  # type: ignore[no-untyped-def]
    class TranslatorMustNotRun:
        def translate(self, image, regions):  # type: ignore[no-untyped-def]
            raise AssertionError("无文字页面不应调用 Qwen")

    input_path = tmp_path / "blank.webp"
    output_path = tmp_path / "blank.translated.png"
    Image.new("RGB", (79, 61), "white").save(input_path)
    result = Pipeline(
        local_config,
        detector=EmptyDetector(),
        translator=TranslatorMustNotRun(),
        inpainter=FakeInpainter(),
        renderer=FakeRenderer(),
        environment_check=lambda: None,
    ).run(input_path, output_path)
    assert result.size == (79, 61)
    assert result.translated_regions == result.skipped_regions == 0
    assert output_path.is_file()


def test_failure_does_not_leave_final_output(tmp_path: Path, local_config) -> None:  # type: ignore[no-untyped-def]
    class FailingRenderer:
        def render(self, image, regions, translations):  # type: ignore[no-untyped-def]
            raise TextLayoutError("region-0001")

    input_path = tmp_path / "source.png"
    output_path = tmp_path / "translated.png"
    Image.new("RGB", (80, 60), "white").save(input_path)
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        with pytest.raises(TextLayoutError):
            Pipeline(
                local_config,
                detector=FakeDetector(),
                translator=FakeTranslator(),
                masker=FakeMasker(),
                inpainter=FakeInpainter(),
                renderer=FailingRenderer(),
                environment_check=lambda: None,
            ).run(input_path, output_path)
    finally:
        logger.remove(sink_id)
    assert not output_path.exists()
    log_text = "".join(messages)
    assert "[7/8] 失败：排版并绘制中文译文" in log_text
    assert "原因：TextLayoutError: region-0001" in log_text


def test_corrupt_input_and_missing_models_fail_clearly(tmp_path: Path, local_config) -> None:  # type: ignore[no-untyped-def]
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"not an image")
    output = tmp_path / "out.png"
    with pytest.raises(ConfigurationError, match="无法读取图片"):
        Pipeline(local_config, environment_check=lambda: None).run(corrupt, output)
    assert not output.exists()

    missing = tmp_path / "missing"
    missing_config = type(local_config)(
        detector_model=missing / "detector",
        qwen_model=missing / "qwen",
        text_mask_model=missing / "text-mask.onnx",
        lama_model=missing / "lama.pt",
    )
    valid = tmp_path / "valid.png"
    Image.new("RGB", (2, 2), "white").save(valid)
    with pytest.raises(ConfigurationError, match="缺少本地模型"):
        Pipeline(missing_config, environment_check=lambda: None).run(valid, output)


def test_default_output_and_input_overwrite_guard(tmp_path: Path, local_config) -> None:  # type: ignore[no-untyped-def]
    input_path = tmp_path / "page.jpeg"
    Image.new("RGB", (10, 10), "white").save(input_path)
    assert default_output_path(input_path).name == "page.translated.png"
    with pytest.raises(ConfigurationError, match="禁止覆盖"):
        Pipeline(local_config, environment_check=lambda: None).run(input_path, input_path)
