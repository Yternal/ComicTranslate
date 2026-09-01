from __future__ import annotations

import io
import json
import subprocess

import pytest

from comictranslate.errors import TranslationError
from comictranslate.qwen import QwenServiceManager
import comictranslate.qwen as qwen_module


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


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self._body.read()

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args) -> None:  # type: ignore[no-untyped-def]
        return None


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


def test_external_service_verifies_model_without_starting_process(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    manager = QwenServiceManager(
        "http://localhost:8000/v1",
        None,
        mode="external",
        model_id="qwen-local",
    )
    monkeypatch.setattr(manager, "_external_model_ids", lambda: {"qwen-local"})
    monkeypatch.setattr(
        manager,
        "_start_process",
        lambda: (_ for _ in ()).throw(AssertionError("不应启动进程")),
    )
    manager.ensure_ready()
    manager.close()
    assert not manager.owned


def test_external_models_probe_reads_openai_models_shape(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    manager = QwenServiceManager(
        "http://127.0.0.1:8000/v1",
        None,
        mode="external",
        model_id="qwen-local",
    )
    seen_urls: list[str] = []

    def fake_open(request, timeout):  # type: ignore[no-untyped-def]
        seen_urls.append(request.full_url)
        assert timeout == 5.0
        return FakeResponse({"data": [{"id": "qwen-local"}, {"id": "other"}]})

    monkeypatch.setattr(qwen_module, "_open", fake_open)
    assert manager._external_model_ids() == {"qwen-local", "other"}
    assert seen_urls == ["http://127.0.0.1:8000/v1/models"]


def test_external_service_rejects_missing_model(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    manager = QwenServiceManager(
        "http://127.0.0.1:8000/v1",
        None,
        mode="external",
        model_id="missing",
    )
    monkeypatch.setattr(manager, "_external_model_ids", lambda: {"available"})
    with pytest.raises(TranslationError, match="找不到模型 missing"):
        manager.ensure_ready()


def test_external_service_rejects_non_loopback_url(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    manager = QwenServiceManager(
        "http://192.168.1.5:8000/v1",
        None,
        mode="external",
        model_id="qwen-local",
    )
    monkeypatch.setattr(
        manager,
        "_external_model_ids",
        lambda: (_ for _ in ()).throw(AssertionError("不应探测远程服务")),
    )
    with pytest.raises(TranslationError, match="只允许本机回环地址"):
        manager.ensure_ready()


def test_external_service_reports_connection_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    manager = QwenServiceManager(
        "http://[::1]:8000/v1",
        None,
        mode="external",
        model_id="qwen-local",
    )

    def fail():  # type: ignore[no-untyped-def]
        raise TranslationError("无法连接外部 Qwen 服务: refused")

    monkeypatch.setattr(manager, "_external_model_ids", fail)
    with pytest.raises(TranslationError, match="refused"):
        manager.ensure_ready()
