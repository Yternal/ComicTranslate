from pathlib import Path

import pytest

from comictranslate.cli import absolute_path, build_parser
from comictranslate.config import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_TEXT_MASK_MODEL,
    PipelineConfig,
)


def test_cli_accepts_original_text_placement() -> None:
    args = build_parser().parse_args(["page.png", "--text-placement", "original"])
    assert args.text_placement == "original"


def test_cli_accepts_device_and_external_qwen_options() -> None:
    args = build_parser().parse_args(
        [
            "page.png",
            "--device",
            "cuda",
            "--qwen-service-mode",
            "external",
            "--qwen-model-id",
            "qwen-vl-local",
        ]
    )
    assert args.device == "cuda"
    assert args.qwen_service_mode == "external"
    assert args.qwen_model_id == "qwen-vl-local"


def test_pipeline_config_rejects_unknown_text_placement() -> None:
    with pytest.raises(ValueError, match="text_placement"):
        PipelineConfig(text_placement="unknown")  # type: ignore[arg-type]


def test_pipeline_config_translates_one_roi_per_request_by_default() -> None:
    assert PipelineConfig().translation_batch_size == 1


def test_cli_accepts_independent_model_paths(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "page.png",
            "--detector-model",
            str(tmp_path / "detector"),
            "--qwen-model",
            str(tmp_path / "qwen"),
            "--text-mask-model",
            str(tmp_path / "text-mask.onnx"),
            "--lama-model",
            str(tmp_path / "big-lama.pt"),
        ]
    )
    assert args.detector_model == (tmp_path / "detector")
    assert args.qwen_model == (tmp_path / "qwen")
    assert args.text_mask_model == (tmp_path / "text-mask.onnx")
    assert args.lama_model == (tmp_path / "big-lama.pt")


def test_independent_model_paths_override_only_their_defaults(tmp_path) -> None:
    config = PipelineConfig(
        detector_model=str(tmp_path / "detector"),
        lama_model=str(tmp_path / "lama.pt"),
    )
    assert config.detector_model == (tmp_path / "detector")
    assert config.qwen_model == DEFAULT_QWEN_MODEL
    assert config.text_mask_model == DEFAULT_TEXT_MASK_MODEL
    assert config.lama_model == (tmp_path / "lama.pt")


def test_cli_rejects_relative_independent_model_path() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["page.png", "--detector-model", "models/detector"]
        )


def test_pipeline_config_rejects_relative_independent_model_path() -> None:
    with pytest.raises(ValueError, match="detector_model 必须是绝对路径"):
        PipelineConfig(detector_model="models/detector")


def test_windows_absolute_model_path_is_accepted_cross_platform() -> None:
    value = r"C:\models\detector"
    assert absolute_path(value) == Path(value)
    assert PipelineConfig(detector_model=value).detector_model == Path(value)
