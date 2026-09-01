from __future__ import annotations

from types import SimpleNamespace

import pytest

from comictranslate.errors import ConfigurationError
from comictranslate.validation import (
    ensure_supported_runtime,
    resolve_device,
    resolve_qwen_service_mode,
)


def _torch(*, cuda: bool, mps: bool):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


@pytest.mark.parametrize(
    ("system", "machine"),
    [("Darwin", "arm64"), ("Windows", "AMD64"), ("Windows", "amd64")],
)
def test_supported_runtime_accepts_target_platforms(system: str, machine: str) -> None:
    ensure_supported_runtime(
        system=system, machine=machine, version_info=(3, 10, 18)
    )


@pytest.mark.parametrize(
    ("system", "machine"),
    [("Linux", "x86_64"), ("Darwin", "x86_64"), ("Windows", "ARM64")],
)
def test_supported_runtime_rejects_other_platforms(system: str, machine: str) -> None:
    with pytest.raises(ConfigurationError, match="仅支持"):
        ensure_supported_runtime(
            system=system, machine=machine, version_info=(3, 10, 18)
        )


def test_supported_runtime_requires_exact_python_version() -> None:
    with pytest.raises(ConfigurationError, match="Python 3.10.18"):
        ensure_supported_runtime(
            system="Windows", machine="AMD64", version_info=(3, 10, 17)
        )


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_priority(cuda: bool, mps: bool, expected: str) -> None:
    assert resolve_device("auto", torch_module=_torch(cuda=cuda, mps=mps)) == expected


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_unavailable_device_fails(device: str) -> None:
    with pytest.raises(ConfigurationError, match="显式指定"):
        resolve_device(device, torch_module=_torch(cuda=False, mps=False))  # type: ignore[arg-type]


def test_qwen_auto_mode_depends_on_platform() -> None:
    assert (
        resolve_qwen_service_mode("auto", system="Darwin", machine="arm64")
        == "managed-mlx"
    )
    assert (
        resolve_qwen_service_mode("auto", system="Windows", machine="AMD64")
        == "external"
    )


def test_managed_mlx_is_rejected_on_windows() -> None:
    with pytest.raises(ConfigurationError, match="仅支持 Apple Silicon"):
        resolve_qwen_service_mode(
            "managed-mlx", system="Windows", machine="AMD64"
        )
