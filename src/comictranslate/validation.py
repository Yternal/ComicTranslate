from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from .config import PipelineConfig
from .errors import ConfigurationError
from .rendering import resolve_font_path


ResolvedDevice = Literal["cpu", "cuda", "mps"]
ResolvedQwenServiceMode = Literal["managed-mlx", "managed-llama", "external"]


def ensure_supported_runtime(
    *,
    system: str | None = None,
    machine: str | None = None,
    version_info: tuple[int, int, int] | None = None,
) -> None:
    actual_version = version_info or sys.version_info[:3]
    if actual_version != (3, 10, 18):
        actual = ".".join(str(item) for item in actual_version)
        raise ConfigurationError(f"必须使用 Python 3.10.18，当前为 {actual}")
    actual_system = system or platform.system()
    actual_machine = machine or platform.machine()
    supported = (actual_system == "Darwin" and actual_machine == "arm64") or (
        actual_system == "Windows" and actual_machine.upper() == "AMD64"
    )
    if not supported:
        raise ConfigurationError(
            "仅支持 Apple Silicon macOS 或 Windows x64，"
            f"当前为 {actual_system}/{actual_machine}"
        )


def resolve_qwen_service_mode(
    requested: Literal["auto", "managed-mlx", "managed-llama", "external"],
    *,
    system: str | None = None,
    machine: str | None = None,
    model_path: Path | None = None,
    server_executable: Path | None = None,
    model_id: str | None = None,
) -> ResolvedQwenServiceMode:
    actual_system = system or platform.system()
    actual_machine = machine or platform.machine()
    if requested == "auto":
        if actual_system == "Darwin" and actual_machine == "arm64":
            return "managed-mlx"
        if model_path is not None or server_executable is not None:
            return "managed-llama"
        if model_id:
            return "external"
        raise ConfigurationError("请配置 Qwen GGUF、qwen_server_executable 和 qwen_mmproj，或外部服务模型 ID")
    if requested == "managed-mlx" and not (
        actual_system == "Darwin" and actual_machine == "arm64"
    ):
        raise ConfigurationError("managed-mlx 模式仅支持 Apple Silicon macOS")
    return requested


def resolve_device(
    requested: Literal["auto", "cpu", "cuda", "mps"],
    *,
    torch_module: Any = None,
) -> ResolvedDevice:
    try:
        torch = torch_module
        if torch is None:
            import torch
    except Exception as exc:
        raise ConfigurationError(f"无法检查 PyTorch 计算设备: {exc}") from exc

    if requested == "cpu":
        return "cpu"
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        if requested == "auto":
            logger.warning("CUDA 初始化失败，自动回退 CPU：{}", exc)
            return "cpu"
        raise ConfigurationError(f"CUDA 初始化失败: {exc}") from exc
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    if cuda_available and requested in {"auto", "cuda"}:
        try:
            probe = torch.ones((2, 2), device="cuda")
            (probe @ probe).sum().item()
            torch.cuda.synchronize()
        except Exception as exc:
            if requested == "cuda":
                raise ConfigurationError(f"CUDA 实际运算失败，请检查驱动和 PyTorch CUDA 构建: {exc}") from exc
            logger.warning("CUDA 实际运算失败，自动回退 CPU：{}", exc)
            return "cpu"
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        logger.warning("CUDA 和 MPS 均不可用，自动回退到 CPU")
        return "cpu"
    if requested == "cuda" and not cuda_available:
        raise ConfigurationError("显式指定了 CUDA，但 torch.cuda.is_available() 为 false")
    if requested == "mps" and not mps_available:
        raise ConfigurationError("显式指定了 MPS，但当前环境不可用")
    return requested


def validate_models_and_font(
    config: PipelineConfig, qwen_service_mode: ResolvedQwenServiceMode
) -> Path:
    required: list[tuple[str, Path | None, bool]] = [
        ("RT-DETR", config.detector_model, True),
        ("文字 mask ONNX", config.text_mask_model, False),
        ("LaMa", config.lama_model, False),
    ]
    if qwen_service_mode == "managed-mlx":
        required.append(("Qwen MLX", config.qwen_model, True))
    elif qwen_service_mode == "managed-llama":
        required.extend([
            ("Qwen GGUF", config.qwen_model, False),
            ("llama-server", config.qwen_server_executable, False),
            ("Qwen mmproj", config.qwen_mmproj, False),
        ])
    elif not config.qwen_model_id:
        raise ConfigurationError("external Qwen 模式必须配置 qwen_model_id")
    missing = []
    for name, path, is_directory in required:
        if path is None:
            missing.append(f"{name}: 未配置")
        elif not (path.is_dir() if is_directory else path.is_file()):
            missing.append(f"{name}: {path}")
    if missing:
        raise ConfigurationError("缺少本地模型：" + "；".join(missing))
    return resolve_font_path(config.font_path)
