from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .errors import InpaintingError


class LamaInpainter:
    def __init__(self, model_path: Path, *, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._model: Any = None
        self._torch: Any = None
        self._device: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch

            device = torch.device(self.device)
            model = torch.jit.load(str(self.model_path), map_location=device)
            model.eval()
        except Exception as exc:
            raise InpaintingError(f"无法加载 LaMa 模型 {self.model_path}: {exc}") from exc
        self._torch = torch
        self._device = device
        self._model = model

    @staticmethod
    def _pad(array: np.ndarray, target_height: int, target_width: int, *, mask: bool) -> np.ndarray:
        bottom = target_height - array.shape[0]
        right = target_width - array.shape[1]
        if mask:
            return np.pad(array, ((0, bottom), (0, right)), mode="constant")
        mode = "reflect" if array.shape[0] > 1 and array.shape[1] > 1 else "edge"
        return np.pad(array, ((0, bottom), (0, right), (0, 0)), mode=mode)

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if mask.shape != rgb.shape[:2]:
            raise InpaintingError(
                f"全局 mask 尺寸 {mask.shape} 与图片尺寸 {rgb.shape[:2]} 不一致"
            )
        if not np.any(mask > 127):
            return Image.fromarray(rgb, mode="RGB")
        self._load()
        height, width = rgb.shape[:2]
        padded_height = ((height + 7) // 8) * 8
        padded_width = ((width + 7) // 8) * 8
        image_array = self._pad(rgb, padded_height, padded_width, mask=False)
        mask_array = self._pad(
            np.where(mask > 127, 1.0, 0.0).astype(np.float32),
            padded_height,
            padded_width,
            mask=True,
        )
        image_tensor = (
            self._torch.from_numpy(image_array.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device)
        )
        mask_tensor = (
            self._torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(self._device)
        )
        try:
            with self._torch.inference_mode():
                result = self._model(image_tensor, mask_tensor)
        except Exception as exc:
            raise InpaintingError(f"LaMa 推理失败: {exc}") from exc
        result_array = (
            result[0]
            .permute(1, 2, 0)
            .detach()
            .to("cpu")
            .clamp(0, 1)
            .numpy()[:height, :width]
        )
        if result_array.shape[:2] != (height, width):
            raise InpaintingError("LaMa 裁剪后尺寸与输入不一致")
        return Image.fromarray(np.rint(result_array * 255).astype(np.uint8), mode="RGB")
