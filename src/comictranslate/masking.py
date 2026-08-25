from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .errors import MaskError
from .models import Region
from .vendor.comic_text_detector import refine_mask


class ComicTextMasker:
    def __init__(self, model_path: Path, input_size: int = 1024) -> None:
        self.model_path = Path(model_path)
        self.input_size = input_size
        self._network: Any = None
        self._output_names: tuple[str, ...] = ()

    def _load(self) -> None:
        if self._network is not None:
            return
        try:
            network = cv2.dnn.readNetFromONNX(str(self.model_path))
            output_names = tuple(network.getUnconnectedOutLayersNames())
        except Exception as exc:
            raise MaskError(f"无法加载文字 mask ONNX 模型 {self.model_path}: {exc}") from exc
        self._network = network
        self._output_names = output_names

    def _raw_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        self._load()
        height, width = image_bgr.shape[:2]
        ratio = min(self.input_size / height, self.input_size / width)
        resized_width = max(1, round(width * ratio))
        resized_height = max(1, round(height * ratio))
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            0,
            self.input_size - resized_height,
            0,
            self.input_size - resized_width,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
        )
        try:
            self._network.setInput(blob)
            values = self._network.forward(self._output_names)
        except Exception as exc:
            raise MaskError(f"文字 mask ONNX 推理失败: {exc}") from exc
        outputs = dict(zip(self._output_names, values))
        segmentation = outputs.get("seg")
        if segmentation is None:
            candidates = [value for value in values if value.ndim == 4 and value.shape[1] == 1]
            if not candidates:
                raise MaskError(f"ONNX 输出中找不到 seg，实际输出为 {self._output_names}")
            segmentation = candidates[0]
        soft_mask = np.squeeze(segmentation)
        if soft_mask.ndim != 2:
            raise MaskError(f"seg 输出维度无效: {segmentation.shape}")
        soft_mask = soft_mask[:resized_height, :resized_width]
        soft_mask = cv2.resize(soft_mask, (width, height), interpolation=cv2.INTER_LINEAR)
        if soft_mask.max(initial=0) <= 1.0:
            soft_mask = soft_mask * 255.0
        return np.clip(soft_mask, 0, 255).astype(np.uint8)

    def mask(self, roi: Image.Image) -> np.ndarray:
        rgb = np.asarray(roi.convert("RGB"), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        raw = self._raw_mask(bgr)
        return refine_mask(bgr, raw)


def stitch_masks(
    image_size: tuple[int, int],
    regions: Sequence[Region],
    roi_masks: Mapping[str, np.ndarray],
) -> np.ndarray:
    width, height = image_size
    global_mask = np.zeros((height, width), np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for region in regions:
        if region.id not in roi_masks:
            raise MaskError(f"区域 {region.id} 缺少 ROI mask")
        x1, y1, x2, y2 = region.crop_bbox.as_int()
        expected_shape = (y2 - y1, x2 - x1)
        roi_mask = np.asarray(roi_masks[region.id])
        if roi_mask.shape != expected_shape:
            raise MaskError(
                f"区域 {region.id} mask 尺寸 {roi_mask.shape} 与 ROI {expected_shape} 不一致"
            )
        binary = np.where(roi_mask > 127, 255, 0).astype(np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        target = global_mask[y1:y2, x1:x2]
        global_mask[y1:y2, x1:x2] = cv2.bitwise_or(target, binary)
    return global_mask
