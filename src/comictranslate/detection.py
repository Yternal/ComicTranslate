from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .errors import DetectionError
from .models import BBox, Detection


def _patch_rtdetr_v2_mps(torch: Any) -> None:
    """Avoid float64 tensors, which PyTorch MPS cannot execute."""
    try:
        from transformers.models.rt_detr_v2 import modeling_rt_detr_v2
    except ImportError:
        return

    def build_position_embedding(
        height: int,
        width: int,
        embed_dim: int = 256,
        temperature: float = 10000.0,
        cls_token: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> Any:
        if embed_dim % 4:
            raise ValueError(f"embed_dim 必须能被 4 整除，实际为 {embed_dim}")
        dtype = dtype or torch.float32
        pos_dim = embed_dim // 4
        device_type = torch.device(device).type if device is not None else "cpu"
        compute_dtype = torch.float32 if device_type == "mps" else torch.float64
        omega = torch.arange(pos_dim, dtype=compute_dtype, device=device) / pos_dim
        omega = 1.0 / temperature**omega
        grid_h = torch.arange(height, dtype=compute_dtype, device=device)
        grid_w = torch.arange(width, dtype=compute_dtype, device=device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        embedding_h = grid_h.flatten().outer(omega)
        embedding_w = grid_w.flatten().outer(omega)
        result = torch.cat(
            [
                embedding_h.sin(),
                embedding_h.cos(),
                embedding_w.sin(),
                embedding_w.cos(),
            ],
            dim=1,
        )
        if cls_token:
            result = torch.cat(
                [torch.zeros(1, embed_dim, dtype=compute_dtype, device=device), result],
                dim=0,
            )
        return result.to(dtype)

    modeling_rt_detr_v2.build_2d_sinusoidal_position_embedding = build_position_embedding


class RTDetrDetector:
    def __init__(self, model_path: Path, threshold: float = 0.3) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._processor: Any = None
        self._model: Any = None
        self._device: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

            if torch.backends.mps.is_available():
                _patch_rtdetr_v2_mps(torch)
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            processor = RTDetrImageProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )
            model = RTDetrV2ForObjectDetection.from_pretrained(
                self.model_path, local_files_only=True
            ).to(device)
            model.eval()
        except Exception as exc:
            raise DetectionError(f"无法加载 RT-DETR 模型 {self.model_path}: {exc}") from exc
        self._processor = processor
        self._model = model
        self._device = device

    def _infer(self, image: Image.Image, device: Any) -> dict[str, Any]:
        import torch

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([[image.height, image.width]], device=device)
        return self._processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
        )[0]

    def detect(self, image: Image.Image) -> list[Detection]:
        self._load()
        try:
            results = self._infer(image.convert("RGB"), self._device)
        except RuntimeError as exc:
            if getattr(self._device, "type", None) != "mps":
                raise DetectionError(f"RT-DETR 推理失败: {exc}") from exc
            try:
                import torch

                self._device = torch.device("cpu")
                self._model = self._model.to(self._device)
                results = self._infer(image.convert("RGB"), self._device)
            except Exception as cpu_exc:
                raise DetectionError(
                    f"RT-DETR 在 MPS 和 CPU 上均推理失败: {cpu_exc}"
                ) from cpu_exc
        except Exception as exc:
            raise DetectionError(f"RT-DETR 推理失败: {exc}") from exc

        detections: list[Detection] = []
        allowed = {"bubble", "text_bubble", "text_free"}
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            numeric_label = int(label_id.detach().cpu())
            label = self._model.config.id2label[numeric_label]
            if label not in allowed:
                continue
            x1, y1, x2, y2 = box.detach().cpu().tolist()
            detections.append(
                Detection(
                    label=label,
                    score=float(score.detach().cpu()),
                    bbox=BBox(x1, y1, x2, y2),
                )
            )
        return detections
