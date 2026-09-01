from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from comictranslate.errors import ConfigurationError
from comictranslate.validation import validate_models_and_font


def test_external_mode_requires_model_id_but_not_local_qwen(
    local_config,
) -> None:  # type: ignore[no-untyped-def]
    config = replace(local_config, qwen_model=None, qwen_model_id="qwen-local")
    assert validate_models_and_font(config, "external") == config.font_path


def test_external_mode_reports_missing_model_id(local_config) -> None:  # type: ignore[no-untyped-def]
    config = replace(local_config, qwen_model=None, qwen_model_id=None)
    with pytest.raises(ConfigurationError, match="qwen_model_id"):
        validate_models_and_font(config, "external")


def test_managed_mlx_requires_local_qwen_directory(
    local_config, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    config = replace(local_config, qwen_model=tmp_path / "missing-qwen")
    with pytest.raises(ConfigurationError, match="Qwen MLX"):
        validate_models_and_font(config, "managed-mlx")
