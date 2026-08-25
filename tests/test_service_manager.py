from __future__ import annotations

import subprocess

import pytest

from comictranslate.errors import TranslationError
from comictranslate.qwen import QwenServiceManager


class FakeProcess:
    def __init__(self, returncode=None) -> None:  # type: ignore[no-untyped-def]
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):  # type: ignore[no-untyped-def]
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):  # type: ignore[no-untyped-def]
        return self.returncode


def test_reuses_existing_mlx_service_without_owning_it(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    manager = QwenServiceManager("http://127.0.0.1:8080/v1", "/model")
    monkeypatch.setattr(manager, "_probe", lambda: "mlx")
    manager.ensure_ready()
    assert not manager.owned
    manager.close()


def test_starts_and_only_closes_owned_service(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    states = iter(["unreachable", "mlx"])
    process = FakeProcess()
    manager = QwenServiceManager("http://127.0.0.1:8080/v1", "/model")
    monkeypatch.setattr(manager, "_probe", lambda: next(states))
    monkeypatch.setattr(manager, "_port_is_open", lambda: False)
    monkeypatch.setattr(manager, "_start_process", lambda: process)
    manager.ensure_ready()
    assert manager.owned
    manager.close()
    assert process.terminated


def test_non_mlx_service_on_port_is_rejected(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    manager = QwenServiceManager("http://127.0.0.1:9090/v1", "/model")
    monkeypatch.setattr(manager, "_probe", lambda: "other")
    with pytest.raises(TranslationError, match="非 mlx_vlm"):
        manager.ensure_ready()


def test_open_port_without_health_is_rejected(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    manager = QwenServiceManager("http://127.0.0.1:9090/v1", "/model")
    monkeypatch.setattr(manager, "_probe", lambda: "unreachable")
    monkeypatch.setattr(manager, "_port_is_open", lambda: True)
    with pytest.raises(TranslationError, match="非 mlx_vlm"):
        manager.ensure_ready()


def test_start_timeout_terminates_owned_process(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    now = [0.0]
    process = FakeProcess()
    manager = QwenServiceManager(
        "http://127.0.0.1:8080/v1",
        "/model",
        start_timeout=2,
        clock=lambda: now[0],
        sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    monkeypatch.setattr(manager, "_probe", lambda: "unreachable")
    monkeypatch.setattr(manager, "_port_is_open", lambda: False)
    monkeypatch.setattr(manager, "_start_process", lambda: process)
    with pytest.raises(TranslationError, match="2 秒内未就绪"):
        manager.ensure_ready()
    assert process.terminated


def test_close_never_terminates_non_owned_process() -> None:
    process = FakeProcess()
    manager = QwenServiceManager("http://127.0.0.1:8080/v1", "/model")
    manager._process = process  # type: ignore[assignment]
    manager._owned = False
    manager.close()
    assert not process.terminated
