from pathlib import Path

import pytest

from comictranslate.cli import build_parser
from comictranslate.config import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_TEXT_MASK_MODEL,
    PipelineConfig,
)


def test_cli_accepts_original_text_placement() -> None:
    args = build_parser().parse_args(["page.png", "--text-placement", "original"])
    assert args.text_placement == "original"


def test_pipeline_config_rejects_unknown_text_placement() -> None:
    with pytest.raises(ValueError, match="text_placement"):
        PipelineConfig(text_placement="unknown")  # type: ignore[arg-type]


def test_pipeline_config_translates_one_roi_per_request_by_default() -> None:
    assert PipelineConfig().translation_batch_size == 1


def test_cli_accepts_independent_model_paths() -> None:
    args = build_parser().parse_args(
        [
            "page.png",
            "--detector-model",
            "/models/detector",
            "--qwen-model",
            "/models/qwen",
            "--text-mask-model",
            "/models/text-mask.onnx",
            "--lama-model",
            "/models/big-lama.pt",
        ]
    )
    assert args.detector_model == Path("/models/detector")
    assert args.qwen_model == Path("/models/qwen")
    assert args.text_mask_model == Path("/models/text-mask.onnx")
    assert args.lama_model == Path("/models/big-lama.pt")


def test_independent_model_paths_override_only_their_defaults() -> None:
    config = PipelineConfig(
        detector_model="/custom/detector",
        lama_model="/custom/lama.pt",
    )
    assert config.detector_model == Path("/custom/detector")
    assert config.qwen_model == DEFAULT_QWEN_MODEL
    assert config.text_mask_model == DEFAULT_TEXT_MASK_MODEL
    assert config.lama_model == Path("/custom/lama.pt")


def test_cli_rejects_relative_independent_model_path() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["page.png", "--detector-model", "models/detector"]
        )


def test_pipeline_config_rejects_relative_independent_model_path() -> None:
    with pytest.raises(ValueError, match="detector_model 必须是绝对路径"):
        PipelineConfig(detector_model="models/detector")
