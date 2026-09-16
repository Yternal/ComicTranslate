from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from comictranslate.config import PipelineConfig
from comictranslate.errors import ConfigurationError, TranslationError
from comictranslate.io_utils import atomic_save
from comictranslate.qwen import QwenServiceManager, QwenTranslator
from comictranslate.validation import resolve_device, resolve_qwen_service_mode
from test_service_manager import FakeProcess


def test_windows_auto_requires_configuration(tmp_path):
    options = dict(system="Windows", machine="AMD64")
    with pytest.raises(ConfigurationError, match="请配置"):
        resolve_qwen_service_mode("auto", **options)
    assert resolve_qwen_service_mode("auto", model_id="qwen", **options) == "external"
    assert resolve_qwen_service_mode("auto", model_path=tmp_path / "q.gguf", **options) == "managed-llama"
    assert resolve_qwen_service_mode("external", model_path=tmp_path / "q.gguf", **options) == "external"


@pytest.mark.parametrize("layers", ["-1", "1.5", "all", ""])
def test_invalid_gpu_layers(layers):
    with pytest.raises(ValueError, match="qwen_gpu_layers"):
        PipelineConfig(qwen_gpu_layers=layers)


def test_cuda_operation_failure_falls_back_only_in_auto():
    def fail(*args, **kwargs):
        raise RuntimeError("no kernel image")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), ones=fail)
    assert resolve_device("auto", torch_module=torch) == "cpu"
    with pytest.raises(ConfigurationError, match="实际运算失败"):
        resolve_device("cuda", torch_module=torch)


def test_interrupt_during_start_cleans_owned_process(monkeypatch, tmp_path):
    process = FakeProcess()
    def interrupt(_seconds):
        raise KeyboardInterrupt
    manager = QwenServiceManager("http://localhost:8080/v1", tmp_path, sleeper=interrupt)
    monkeypatch.setattr(manager, "_probe", lambda: "unreachable")
    monkeypatch.setattr(manager, "_port_is_open", lambda: False)
    monkeypatch.setattr(manager, "_start_process", lambda: process)
    with pytest.raises(KeyboardInterrupt):
        manager.ensure_ready()
    assert process.terminated and not manager.owned


def test_llama_launch_preserves_unicode_paths_and_logs(monkeypatch, tmp_path):
    import comictranslate.qwen as module
    model = tmp_path / "中文 模型.gguf"
    manager = QwenServiceManager(
        "http://127.0.0.1:8080/v1", model, mode="managed-llama",
        server_executable=tmp_path / "程序 目录" / "llama-server.exe",
        mmproj=tmp_path / "视觉 投影.gguf", gpu_layers="0",
    )
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(b"startup diagnostics")
        return FakeProcess()
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    manager._start_process()
    command, kwargs = calls[0]
    assert command[command.index("--model") + 1] == str(model)
    assert command[command.index("--alias") + 1] == model.stem
    assert kwargs["shell"] is False
    assert manager.log_path.read_bytes() == b"startup diagnostics"
    manager.log_path.unlink()


def test_reuse_checks_llama_model_and_vision(monkeypatch, tmp_path):
    model = tmp_path / "q.gguf"
    manager = QwenServiceManager("http://localhost:8080/v1", model, mode="managed-llama")
    monkeypatch.setattr(manager, "_probe", lambda: "llama")
    monkeypatch.setattr(manager, "_external_model_ids", lambda: {"q"})
    props = {"model_path": str(model), "modalities": {"vision": True},
             "default_generation_settings": {"n_ctx": 32768}}
    monkeypatch.setattr(manager, "_properties", lambda: props)
    manager.ensure_ready()
    assert not manager.owned
    props["model_path"] = str(tmp_path / "other.gguf")
    with pytest.raises(TranslationError, match="路径"):
        manager.ensure_ready()
    props["model_path"] = str(model)
    props["modalities"]["vision"] = False
    with pytest.raises(TranslationError, match="视觉"):
        manager.ensure_ready()


def test_locked_output_preserves_original_and_removes_temp(monkeypatch, tmp_path):
    import comictranslate.io_utils as module
    output = tmp_path / "中文 页面.png"
    output.write_bytes(b"original")
    def locked(*args):
        raise PermissionError("file in use")
    monkeypatch.setattr(module.os, "replace", locked)
    with pytest.raises(ConfigurationError, match="file in use"):
        atomic_save(Image.new("RGB", (4, 4)), output)
    assert output.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [output]


def test_onnx_load_uses_bytes_for_unicode_path(monkeypatch, tmp_path):
    import comictranslate.masking as module
    path = tmp_path / "中文 模型.onnx"
    path.write_bytes(b"onnx bytes")
    received = []
    def read(data):
        received.append(data.tobytes())
        return SimpleNamespace(getUnconnectedOutLayersNames=lambda: ["mask"])
    monkeypatch.setattr(module.cv2.dnn, "readNetFromONNX", read)
    module.ComicTextMasker(path)._load()
    assert received == [b"onnx bytes"]


def test_cli_reconfigures_redirected_stdout_as_utf8(monkeypatch, tmp_path):
    import comictranslate.cli as module
    from comictranslate.models import PipelineResult
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii")
    monkeypatch.setattr(module.sys, "stdout", stream)
    monkeypatch.setattr(module, "translate_image", lambda *a: PipelineResult(4, 4, 0, 0, tmp_path / "中文.png"))
    assert module.main([str(tmp_path / "输入.png")]) == 0
    stream.flush()
    assert "中文.png" in json.loads(raw.getvalue().decode("utf-8"))["output_path"]


def test_llama_http_multimodal_schema_contract():
    from comictranslate.models import Region, BBox
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            content = json.dumps({"items": [{"id": "r1", "source_text": "你好", "action": "skip", "translation": ""}]})
            body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        translator = QwenTranslator(f"http://127.0.0.1:{server.server_port}/v1", "q", service_mode="managed-llama")
        region = Region("r1", "free", BBox(0, 0, 40, 40), BBox(0, 0, 40, 40), None)
        result = translator.translate(Image.new("RGB", (80, 80)), [region])
        assert result[0].action == "skip"
        path, payload = requests[0]
        assert path == "/v1/chat/completions"
        assert len([c for c in payload["messages"][0]["content"] if c["type"] == "image_url"]) == 2
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert "enable_thinking" not in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cuda_initialization_failure_falls_back(monkeypatch):
    def fail():
        raise RuntimeError("driver initialization failed")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=fail))
    assert resolve_device("auto", torch_module=torch) == "cpu"
    assert resolve_device("cpu", torch_module=torch) == "cpu"
    with pytest.raises(ConfigurationError, match="初始化失败"):
        resolve_device("cuda", torch_module=torch)


def test_managed_llama_validates_all_local_files(tmp_path, local_config):
    from dataclasses import replace
    from comictranslate.validation import validate_models_and_font
    model, projector, executable = [tmp_path / name for name in ("q.gguf", "mmproj.gguf", "llama-server.exe")]
    config = replace(local_config, qwen_model=model, qwen_mmproj=projector, qwen_server_executable=executable)
    with pytest.raises(ConfigurationError, match="Qwen GGUF.*llama-server.*Qwen mmproj"):
        validate_models_and_font(config, "managed-llama")
    for path in (model, projector, executable):
        path.touch()
    assert validate_models_and_font(config, "managed-llama") == config.font_path


def test_llama_cli_options_reach_pipeline(monkeypatch, tmp_path):
    import comictranslate.cli as module
    from comictranslate.models import PipelineResult
    captured = []
    def translate(source, output, config):
        captured.append(config)
        return PipelineResult(1, 1, 0, 0, output)
    monkeypatch.setattr(module, "translate_image", translate)
    assert module.main([str(tmp_path / "page.png"), "--qwen-service-mode", "managed-llama",
                        "--qwen-model", str(tmp_path / "q.gguf"),
                        "--qwen-mmproj", str(tmp_path / "投影.gguf"),
                        "--qwen-server-executable", str(tmp_path / "llama-server.exe"),
                        "--qwen-context-size", "16384", "--qwen-gpu-layers", "0"]) == 0
    assert captured[0].qwen_context_size == 16384
    assert captured[0].qwen_gpu_layers == "0"
    assert captured[0].qwen_mmproj == tmp_path / "投影.gguf"
