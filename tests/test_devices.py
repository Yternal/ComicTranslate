from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from comictranslate.detection import RTDetrDetector
from comictranslate.errors import DetectionError
from comictranslate.inpainting import LamaInpainter


class MovableModel:
    def __init__(self) -> None:
        self.moves: list[str] = []
        self.config = SimpleNamespace(id2label={})

    def to(self, device):  # type: ignore[no-untyped-def]
        self.moves.append(device.type)
        return self


def test_rtdetr_retries_mps_runtime_error_on_cpu(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    detector = RTDetrDetector("/model", device="mps")
    detector._model = MovableModel()
    detector._processor = object()
    detector._device = SimpleNamespace(type="mps")
    calls: list[str] = []

    def infer(_image, device):  # type: ignore[no-untyped-def]
        calls.append(device.type)
        if device.type == "mps":
            raise RuntimeError("unsupported MPS op")
        return {"scores": [], "labels": [], "boxes": []}

    monkeypatch.setattr(detector, "_infer", infer)
    assert detector.detect(Image.new("RGB", (4, 4))) == []
    assert calls == ["mps", "cpu"]
    assert detector._model.moves == ["cpu"]


def test_rtdetr_cuda_runtime_error_does_not_fallback(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    detector = RTDetrDetector("/model", device="cuda")
    detector._model = MovableModel()
    detector._processor = object()
    detector._device = SimpleNamespace(type="cuda")
    monkeypatch.setattr(
        detector,
        "_infer",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("CUDA failure")),
    )

    with pytest.raises(DetectionError, match="CUDA failure"):
        detector.detect(Image.new("RGB", (4, 4)))
    assert detector._model.moves == []


def test_lama_keeps_requested_device_until_model_load() -> None:
    assert LamaInpainter("/model", device="cuda").device == "cuda"
