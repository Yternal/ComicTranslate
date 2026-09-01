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
from .errors import ComicTranslateError, ConfigurationError, TranslationError
from .geometry import build_regions
from .inpainting import LamaInpainter
from .io_utils import (
    atomic_save,
    default_output_directory,
    directory_output_path,
    discover_images,
    load_image,
    original_alpha,
    restore_alpha,
    validate_io_paths,
)
from .masking import ComicTextMasker, stitch_masks
from .models import BatchFailure, BatchResult, PipelineResult, Region, Translation
from .qwen import QwenServiceManager, QwenTranslator
from .rendering import ChineseTextRenderer
from .validation import (
    ResolvedQwenServiceMode,
    ensure_supported_runtime,
    resolve_device,
    resolve_qwen_service_mode,
    validate_models_and_font,
)


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
        environment_check: Callable[[], None] = ensure_supported_runtime,
        device_resolver: Callable[[str], str] = resolve_device,
    ) -> None:
        self.config = config
        self.detector = detector
        self.translator = translator
        self.masker = masker
        self.inpainter = inpainter
        self.renderer = renderer
        self.environment_check = environment_check
        self.device_resolver = device_resolver
        self._font_path: Path | None = None
        self._device: str | None = None
        self._qwen_service_mode: ResolvedQwenServiceMode | None = None
        self._prepared = False

    def prepare(self) -> None:
        if self._prepared:
            return
        self.environment_check()
        service_mode = resolve_qwen_service_mode(self.config.qwen_service_mode)
        device = self.device_resolver(self.config.device)
        font_path = validate_models_and_font(self.config, service_mode)
        self._font_path = font_path
        self._device = device
        self._qwen_service_mode = service_mode
        self._prepared = True

    def _resolved_qwen_model(self) -> str | Path:
        if self._qwen_service_mode == "external":
            if self.config.qwen_model_id is None:
                raise RuntimeError("管线尚未完成 Qwen 外部模型初始化")
            return self.config.qwen_model_id
        if self.config.qwen_model is None:
            raise RuntimeError("管线尚未完成 Qwen MLX 模型初始化")
        return self.config.qwen_model

    def _service_manager(self) -> QwenServiceManager:
        if self._qwen_service_mode is None:
            raise RuntimeError("管线尚未完成 Qwen 服务模式初始化")
        return QwenServiceManager(
            self.config.server_url,
            self.config.qwen_model,
            mode=self._qwen_service_mode,
            model_id=self.config.qwen_model_id,
            start_timeout=self.config.server_start_timeout,
        )

    def _translator(self) -> QwenTranslator:
        if self._qwen_service_mode is None:
            raise RuntimeError("管线尚未完成 Qwen 服务模式初始化")
        return QwenTranslator(
            self.config.server_url,
            self._resolved_qwen_model(),
            service_mode=self._qwen_service_mode,
            batch_size=self.config.translation_batch_size,
        )

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

    def _debug_json(self, debug_dir: Path | None, name: str, payload: object) -> None:
        if debug_dir is None:
            return
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _prepare_debug_dir(self, debug_dir: Path | None) -> None:
        if debug_dir is None:
            return
        debug_dir.mkdir(parents=True, exist_ok=True)
        for name in ("detections.json", "translations.json", "mask.png", "clean.png"):
            (debug_dir / name).unlink(missing_ok=True)
        roi_dir = debug_dir / "roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        for stale_roi in roi_dir.glob("region-*.png"):
            stale_roi.unlink()

    def _debug_regions(
        self, debug_dir: Path | None, image: Image.Image, regions: list[Region]
    ) -> None:
        if debug_dir is None:
            return
        roi_dir = debug_dir / "roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        for region in regions:
            image.crop(region.crop_bbox.as_int()).save(roi_dir / f"{region.id}.png")

    def _translate(self, image: Image.Image, regions: list[Region]) -> list[Translation]:
        if not regions:
            return []
        if self.translator is not None:
            return list(self.translator.translate(image, regions))
        translator = self._translator()
        with self._service_manager():
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

    def run(
        self,
        input_path: Path,
        output_path: Path,
        *,
        debug_name: str | None = None,
    ) -> PipelineResult:
        logger.info("开始处理：{} -> {}", input_path, output_path)
        debug_dir = self.config.debug_dir
        if debug_dir is not None and debug_name is not None:
            debug_dir = debug_dir / debug_name
        with self._stage(1, "校验输入、模型与运行环境"):
            input_path, output_path = validate_io_paths(input_path, output_path)
            self.prepare()
            self._prepare_debug_dir(debug_dir)

        with self._stage(2, "读取输入图片"):
            original = load_image(input_path)
            width, height = original.size
            alpha = original_alpha(original)
            rgb = original.convert("RGB")
            logger.info("图片信息：{}x{}，alpha={}", width, height, alpha is not None)

        with self._stage(3, "检测气泡与文字区域"):
            if self.detector is None:
                if self.config.detector_model is None or self._device is None:
                    raise RuntimeError("管线尚未完成 RT-DETR 初始化")
                self.detector = RTDetrDetector(
                    self.config.detector_model,
                    self.config.detection_threshold,
                    device=self._device,
                )
            raw_detections = self.detector.detect(rgb)
            detections, regions = build_regions(
                raw_detections,
                width,
                height,
                nms_iou_threshold=self.config.nms_iou_threshold,
            )
            self._debug_json(
                debug_dir,
                "detections.json",
                {
                    "image_size": [width, height],
                    "detections": [item.to_dict() for item in detections],
                    "regions": [item.to_dict() for item in regions],
                },
            )
            self._debug_regions(debug_dir, rgb, regions)
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
                debug_dir,
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
                if self.masker is None:
                    if self.config.text_mask_model is None:
                        raise RuntimeError("管线尚未完成文字 mask 模型初始化")
                    self.masker = ComicTextMasker(self.config.text_mask_model)
                for region in translated_regions:
                    roi = rgb.crop(region.crop_bbox.as_int())
                    roi_masks[region.id] = self.masker.mask(roi)
            global_mask = stitch_masks(rgb.size, translated_regions, roi_masks)
            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(global_mask, mode="L").save(debug_dir / "mask.png")
            logger.info("掩膜结果：处理 {} 个文字区域", len(roi_masks))

        with self._stage(6, "LaMa 擦除原文字"):
            if self.inpainter is None:
                if self.config.lama_model is None or self._device is None:
                    raise RuntimeError("管线尚未完成 LaMa 初始化")
                self.inpainter = LamaInpainter(
                    self.config.lama_model, device=self._device
                )
            clean = self.inpainter.inpaint(rgb, global_mask)
            if clean.size != rgb.size:
                raise RuntimeError(f"LaMa 输出尺寸 {clean.size} 与输入 {rgb.size} 不一致")
            if debug_dir is not None:
                clean.save(debug_dir / "clean.png")

        with self._stage(7, "排版并绘制中文译文"):
            if self.renderer is None:
                if self._font_path is None:
                    raise RuntimeError("管线尚未完成初始化")
                self.renderer = ChineseTextRenderer(
                    self._font_path,
                    minimum_size=self.config.minimum_font_size,
                    maximum_size=self.config.maximum_font_size,
                    text_placement=self.config.text_placement,
                )
            rendered = self.renderer.render(clean, regions, translations)
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


def translate_directory(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    config: PipelineConfig | None = None,
) -> BatchResult:
    """Translate supported images directly inside one directory without recursion."""
    batch_config = config or PipelineConfig()
    resolved_input_dir = Path(input_dir).expanduser()
    images = discover_images(resolved_input_dir)
    resolved_input_dir = resolved_input_dir.resolve()
    resolved_output_dir = Path(
        output_dir
        if output_dir is not None
        else default_output_directory(resolved_input_dir)
    ).expanduser().resolve(strict=False)
    if resolved_output_dir.exists() and not resolved_output_dir.is_dir():
        raise ConfigurationError(f"批量输出路径不是文件夹: {resolved_output_dir}")

    jobs = tuple(
        (
            input_path.resolve(),
            directory_output_path(input_path, resolved_output_dir),
        )
        for input_path in images
    )
    targets_by_name: dict[str, list[Path]] = {}
    for input_path, output_path in jobs:
        targets_by_name.setdefault(output_path.name.casefold(), []).append(input_path)
    conflicts = [paths for paths in targets_by_name.values() if len(paths) > 1]
    if conflicts:
        details = "；".join(
            f"{', '.join(path.name for path in paths)} -> "
            f"{directory_output_path(paths[0], resolved_output_dir).name}"
            for paths in conflicts
        )
        raise ConfigurationError(f"多个输入图片会写入同一输出文件: {details}")

    skipped_inputs = tuple(
        input_path for input_path, output_path in jobs if output_path.is_file()
    )
    pending_jobs = tuple(
        (input_path, output_path)
        for input_path, output_path in jobs
        if not output_path.is_file()
    )
    for input_path in skipped_inputs:
        logger.info("跳过已有输出：{}", input_path)
    if not pending_jobs:
        return BatchResult(
            input_dir=resolved_input_dir,
            output_dir=resolved_output_dir,
            skipped_inputs=skipped_inputs,
        )

    try:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(
            f"无法创建批量输出文件夹 {resolved_output_dir}: {exc}"
        ) from exc

    service_mode = resolve_qwen_service_mode(batch_config.qwen_service_mode)
    qwen_model: str | Path | None = (
        batch_config.qwen_model_id
        if service_mode == "external"
        else batch_config.qwen_model
    )
    if qwen_model is None:
        qwen_model = ""
    translator = QwenTranslator(
        batch_config.server_url,
        qwen_model,
        service_mode=service_mode,
        batch_size=batch_config.translation_batch_size,
    )
    pipeline = Pipeline(batch_config, translator=translator)
    pipeline.prepare()
    results: list[PipelineResult] = []
    failures: list[BatchFailure] = []
    with QwenServiceManager(
        batch_config.server_url,
        batch_config.qwen_model,
        mode=service_mode,
        model_id=batch_config.qwen_model_id,
        start_timeout=batch_config.server_start_timeout,
    ):
        for index, (input_path, output_path) in enumerate(pending_jobs, start=1):
            logger.info(
                "批量进度 [{}/{}]：{}", index, len(pending_jobs), input_path.name
            )
            try:
                results.append(
                    pipeline.run(
                        input_path,
                        output_path,
                        debug_name=input_path.name,
                    )
                )
            except ComicTranslateError as exc:
                logger.error("图片处理失败，继续批次：{}｜{}", input_path, exc)
                failures.append(
                    BatchFailure(
                        input_path=input_path,
                        output_path=output_path,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
            except Exception as exc:
                logger.exception("图片处理失败，继续批次：{}", input_path)
                failures.append(
                    BatchFailure(
                        input_path=input_path,
                        output_path=output_path,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )

    result = BatchResult(
        input_dir=resolved_input_dir,
        output_dir=resolved_output_dir,
        results=tuple(results),
        skipped_inputs=skipped_inputs,
        failures=tuple(failures),
    )
    logger.info(
        "批量处理完成：成功 {}，跳过 {}，失败 {}",
        result.succeeded,
        result.skipped,
        result.failed,
    )
    return result
