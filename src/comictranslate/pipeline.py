from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image
from loguru import logger

from .config import PipelineConfig
from .detection import RTDetrDetector
from .errors import TranslationError
from .geometry import build_regions
from .inpainting import LamaInpainter
from .io_utils import atomic_save, load_image, original_alpha, restore_alpha, validate_io_paths
from .masking import ComicTextMasker, stitch_masks
from .models import PipelineResult, Region, Translation
from .qwen import QwenServiceManager, QwenTranslator
from .rendering import ChineseTextRenderer
from .validation import ensure_apple_metal, validate_models_and_font


class Pipeline:
    TOTAL_STAGES = 8

    def __init__(
        self,
        config: PipelineConfig,
        *,
        detector: Any = None,
        translator: Any = None,
        masker: Any = None,
        inpainter: Any = None,
        renderer: Any = None,
        environment_check: Callable[[], None] = ensure_apple_metal,
    ) -> None:
        self.config = config
        self.detector = detector
        self.translator = translator
        self.masker = masker
        self.inpainter = inpainter
        self.renderer = renderer
        self.environment_check = environment_check

    @contextmanager
    def _stage(self, number: int, name: str) -> Iterator[None]:
        started_at = perf_counter()
        logger.info("[{}/{}] 开始：{}", number, self.TOTAL_STAGES, name)
        try:
            yield
        except Exception as exc:
            logger.error(
                "[{}/{}] 失败：{}（耗时 {:.2f} 秒）｜原因：{}: {}",
                number,
                self.TOTAL_STAGES,
                name,
                perf_counter() - started_at,
                type(exc).__name__,
                exc,
            )
            raise
        else:
            logger.success(
                "[{}/{}] 完成：{}（耗时 {:.2f} 秒）",
                number,
                self.TOTAL_STAGES,
                name,
                perf_counter() - started_at,
            )

    def _debug_json(self, name: str, payload: object) -> None:
        if self.config.debug_dir is None:
            return
        self.config.debug_dir.mkdir(parents=True, exist_ok=True)
        (self.config.debug_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _prepare_debug_dir(self) -> None:
        if self.config.debug_dir is None:
            return
        self.config.debug_dir.mkdir(parents=True, exist_ok=True)
        for name in ("detections.json", "translations.json", "mask.png", "clean.png"):
            (self.config.debug_dir / name).unlink(missing_ok=True)
        roi_dir = self.config.debug_dir / "roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        for stale_roi in roi_dir.glob("region-*.png"):
            stale_roi.unlink()

    def _debug_regions(self, image: Image.Image, regions: list[Region]) -> None:
        if self.config.debug_dir is None:
            return
        roi_dir = self.config.debug_dir / "roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        for region in regions:
            image.crop(region.crop_bbox.as_int()).save(roi_dir / f"{region.id}.png")

    def _translate(self, image: Image.Image, regions: list[Region]) -> list[Translation]:
        if not regions:
            return []
        if self.translator is not None:
            return list(self.translator.translate(image, regions))
        translator = QwenTranslator(
            self.config.server_url,
            self.config.qwen_model,
            batch_size=self.config.translation_batch_size,
        )
        with QwenServiceManager(
            self.config.server_url,
            self.config.qwen_model,
            start_timeout=self.config.server_start_timeout,
        ):
            return translator.translate(image, regions)

    @staticmethod
    def _validate_translation_set(
        regions: list[Region], translations: list[Translation]
    ) -> None:
        expected = [region.id for region in regions]
        actual = [translation.id for translation in translations]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise TranslationError(
                f"翻译结果 ID 与检测区域不一致，expected={expected}, actual={actual}"
            )

    def run(self, input_path: Path, output_path: Path) -> PipelineResult:
        logger.info("开始处理：{} -> {}", input_path, output_path)
        with self._stage(1, "校验输入、模型与运行环境"):
            input_path, output_path = validate_io_paths(input_path, output_path)
            font_path = validate_models_and_font(self.config)
            self.environment_check()
            self._prepare_debug_dir()

        with self._stage(2, "读取输入图片"):
            original = load_image(input_path)
            width, height = original.size
            alpha = original_alpha(original)
            rgb = original.convert("RGB")
            logger.info("图片信息：{}x{}，alpha={}", width, height, alpha is not None)

        with self._stage(3, "检测气泡与文字区域"):
            detector = self.detector or RTDetrDetector(
                self.config.detector_model, self.config.detection_threshold
            )
            raw_detections = detector.detect(rgb)
            detections, regions = build_regions(
                raw_detections,
                width,
                height,
                nms_iou_threshold=self.config.nms_iou_threshold,
            )
            self._debug_json(
                "detections.json",
                {
                    "image_size": [width, height],
                    "detections": [item.to_dict() for item in detections],
                    "regions": [item.to_dict() for item in regions],
                },
            )
            self._debug_regions(rgb, regions)
            logger.info(
                "检测结果：原始 {} 个，去重后 {} 个，生成 {} 个文字区域",
                len(raw_detections),
                len(detections),
                len(regions),
            )

        with self._stage(4, "OCR 与中文翻译"):
            translations = self._translate(rgb, regions)
            self._validate_translation_set(regions, translations)
            regions_by_id = {region.id: region for region in regions}
            self._debug_json(
                "translations.json",
                [
                    {
                        **item.to_dict(),
                        "roi_file": f"roi/{item.id}.png",
                        "text_bbox": regions_by_id[item.id].text_bbox.to_list(),
                        "crop_bbox": regions_by_id[item.id].crop_bbox.to_list(),
                    }
                    for item in translations
                ],
            )
            translations_by_id = {item.id: item for item in translations}
            translated_regions = [
                region
                for region in regions
                if translations_by_id[region.id].action == "translate"
            ]
            logger.info(
                "翻译结果：{} 个回填，{} 个跳过",
                len(translated_regions),
                len(regions) - len(translated_regions),
            )

        with self._stage(5, "生成文字掩膜"):
            roi_masks: dict[str, np.ndarray] = {}
            if translated_regions:
                masker = self.masker or ComicTextMasker(self.config.text_mask_model)
                for region in translated_regions:
                    roi = rgb.crop(region.crop_bbox.as_int())
                    roi_masks[region.id] = masker.mask(roi)
            global_mask = stitch_masks(rgb.size, translated_regions, roi_masks)
            if self.config.debug_dir is not None:
                self.config.debug_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(global_mask, mode="L").save(
                    self.config.debug_dir / "mask.png"
                )
            logger.info("掩膜结果：处理 {} 个文字区域", len(roi_masks))

        with self._stage(6, "LaMa 擦除原文字"):
            inpainter = self.inpainter or LamaInpainter(self.config.lama_model)
            clean = inpainter.inpaint(rgb, global_mask)
            if clean.size != rgb.size:
                raise RuntimeError(f"LaMa 输出尺寸 {clean.size} 与输入 {rgb.size} 不一致")
            if self.config.debug_dir is not None:
                clean.save(self.config.debug_dir / "clean.png")

        with self._stage(7, "排版并绘制中文译文"):
            renderer = self.renderer or ChineseTextRenderer(
                font_path,
                minimum_size=self.config.minimum_font_size,
                maximum_size=self.config.maximum_font_size,
                text_placement=self.config.text_placement,
            )
            rendered = renderer.render(clean, regions, translations)
            output = restore_alpha(rendered, alpha)
            if output.size != original.size:
                raise RuntimeError(f"输出尺寸 {output.size} 与输入 {original.size} 不一致")

        with self._stage(8, "保存输出图片"):
            atomic_save(output, output_path)

        result = PipelineResult(
            width=width,
            height=height,
            translated_regions=len(translated_regions),
            skipped_regions=len(regions) - len(translated_regions),
            output_path=output_path.resolve(),
            region_ids=tuple(region.id for region in regions),
        )
        logger.success("处理完成：{}", result.output_path)
        return result


def translate_image(
    input_path: str | Path,
    output_path: str | Path,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Translate one comic image. The final file is written only after all stages succeed."""
    return Pipeline(config or PipelineConfig()).run(Path(input_path), Path(output_path))
