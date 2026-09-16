from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import comictranslate.cli as cli_module
import comictranslate.pipeline as pipeline_module
from comictranslate.errors import ConfigurationError, TranslationError
from comictranslate.io_utils import discover_images
from comictranslate.models import BatchFailure, BatchResult, PipelineResult


def test_discover_images_is_non_recursive_filtered_and_sorted(tmp_path: Path) -> None:
    input_dir = tmp_path / "pages"
    input_dir.mkdir()
    (input_dir / "b.JPEG").write_bytes(b"image")
    (input_dir / "A.png").write_bytes(b"image")
    (input_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    nested = input_dir / "chapter"
    nested.mkdir()
    (nested / "nested.webp").write_bytes(b"image")

    images = discover_images(input_dir)

    assert [path.name for path in images] == ["A.png", "b.JPEG"]


def test_discover_images_rejects_missing_and_empty_directories(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="输入文件夹不存在"):
        discover_images(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigurationError, match="没有支持的图片"):
        discover_images(empty)


def test_translate_directory_rejects_output_collisions_before_setup(
    tmp_path: Path, local_config
) -> None:  # type: ignore[no-untyped-def]
    input_dir = tmp_path / "pages"
    input_dir.mkdir()
    (input_dir / "page.jpg").write_bytes(b"image")
    (input_dir / "page.PNG").write_bytes(b"image")

    with pytest.raises(ConfigurationError, match="同一输出文件"):
        pipeline_module.translate_directory(input_dir, config=local_config)

    assert not (input_dir / "translated").exists()


def test_translate_directory_rejects_file_as_output_directory(
    tmp_path: Path, local_config
) -> None:  # type: ignore[no-untyped-def]
    input_dir = tmp_path / "pages"
    input_dir.mkdir()
    (input_dir / "page.png").write_bytes(b"image")
    output_file = tmp_path / "output"
    output_file.write_bytes(b"not a directory")

    with pytest.raises(ConfigurationError, match="不是文件夹"):
        pipeline_module.translate_directory(
            input_dir, output_dir=output_file, config=local_config
        )


def test_translate_directory_skips_existing_outputs_without_setup(
    tmp_path: Path, local_config, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    input_dir = tmp_path / "pages"
    output_dir = input_dir / "translated"
    input_dir.mkdir()
    output_dir.mkdir()
    source = input_dir / "page.jpg"
    output = output_dir / "page.translated.png"
    source.write_bytes(b"source")
    output.write_bytes(b"translated")
    monkeypatch.setattr(
        pipeline_module,
        "Pipeline",
        lambda *_args, **_kwargs: pytest.fail("跳过已有输出时不应初始化管线"),
    )

    result = pipeline_module.translate_directory(input_dir, config=local_config)

    assert result.total == result.skipped == 1
    assert result.succeeded == result.failed == 0
    assert result.skipped_inputs == (source.resolve(),)
    assert output.read_bytes() == b"translated"


def test_translate_directory_continues_after_failure_and_reuses_session(
    tmp_path: Path, local_config, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    input_dir = tmp_path / "pages"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for name in ("01.png", "02.webp", "03.jpg"):
        (input_dir / name).write_bytes(b"image")
    output_dir.mkdir()
    (output_dir / "02.translated.png").write_bytes(b"existing")
    config = replace(local_config, debug_dir=tmp_path / "debug")
    pipeline_instances: list[FakePipeline] = []
    service_events: list[str] = []
    translator_instances: list[object] = []

    class FakeTranslator:
        def __init__(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            translator_instances.append(self)

    class FakePipeline:
        def __init__(self, pipeline_config, *, translator=None) -> None:  # type: ignore[no-untyped-def]
            self.config = pipeline_config
            self.translator = translator
            self.prepared = 0
            self.calls: list[tuple[str, str, str | None]] = []
            pipeline_instances.append(self)

        def prepare(self) -> None:
            self.prepared += 1

        def _translator(self):
            return FakeTranslator()

        def _service_manager(self):
            return FakeServiceManager()

        def run(
            self, input_path: Path, output_path: Path, *, debug_name: str | None = None
        ) -> PipelineResult:
            self.calls.append((input_path.name, output_path.name, debug_name))
            if input_path.name == "03.jpg":
                raise TranslationError("模拟单图失败")
            output_path.write_bytes(b"translated")
            return PipelineResult(10, 20, 1, 0, output_path.resolve())

    class FakeServiceManager:
        def __init__(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            service_events.append("init")

        def __enter__(self):  # type: ignore[no-untyped-def]
            service_events.append("enter")
            return self

        def __exit__(self, *_args) -> None:  # type: ignore[no-untyped-def]
            service_events.append("exit")

    monkeypatch.setattr(pipeline_module, "QwenTranslator", FakeTranslator)
    monkeypatch.setattr(pipeline_module, "Pipeline", FakePipeline)
    monkeypatch.setattr(pipeline_module, "QwenServiceManager", FakeServiceManager)

    result = pipeline_module.translate_directory(
        input_dir, output_dir=output_dir, config=config
    )

    assert len(pipeline_instances) == len(translator_instances) == 1
    assert pipeline_instances[0].prepared == 1
    assert pipeline_instances[0].calls == [
        ("01.png", "01.translated.png", "01.png"),
        ("03.jpg", "03.translated.png", "03.jpg"),
    ]
    assert service_events == ["init", "enter", "exit"]
    assert (result.succeeded, result.skipped, result.failed, result.total) == (1, 1, 1, 3)
    assert result.failures[0].error_type == "TranslationError"
    assert result.failures[0].message == "模拟单图失败"
    payload = result.to_dict()
    assert payload["failed"] == 1
    assert payload["results"][0]["output_path"] == str(
        (output_dir / "01.translated.png").resolve()
    )
    assert payload["failures"][0] == {
        "input_path": str((input_dir / "03.jpg").resolve()),
        "output_path": str((output_dir / "03.translated.png").resolve()),
        "error_type": "TranslationError",
        "message": "模拟单图失败",
    }


@pytest.mark.parametrize(("has_failure", "expected_code"), [(False, 0), (True, 1)])
def test_directory_cli_prints_summary_and_sets_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    has_failure: bool,
    expected_code: int,
) -> None:
    input_dir = tmp_path / "pages"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    failures = (
        BatchFailure(
            input_path=input_dir / "bad.png",
            output_path=output_dir / "bad.translated.png",
            error_type="TranslationError",
            message="failed",
        ),
    ) if has_failure else ()
    batch_result = BatchResult(
        input_dir=input_dir.resolve(),
        output_dir=output_dir.resolve(),
        skipped_inputs=(input_dir / "done.png",),
        failures=failures,
    )
    calls: list[tuple[Path, Path | None]] = []

    def fake_translate_directory(input_path, selected_output, _config):  # type: ignore[no-untyped-def]
        calls.append((input_path, selected_output))
        return batch_result

    monkeypatch.setattr(cli_module, "translate_directory", fake_translate_directory)

    exit_code = cli_module.main([str(input_dir), "-o", str(output_dir)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == expected_code
    assert payload["failed"] == int(has_failure)
    assert calls == [(input_dir, output_dir)]


def test_file_cli_keeps_default_output_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "page.jpg"
    input_path.write_bytes(b"image")
    calls: list[tuple[Path, Path]] = []

    def fake_translate_image(source, output, _config):  # type: ignore[no-untyped-def]
        calls.append((source, output))
        return PipelineResult(10, 20, 1, 0, output.resolve())

    monkeypatch.setattr(cli_module, "translate_image", fake_translate_image)

    exit_code = cli_module.main([str(input_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["output_path"] == str((tmp_path / "page.translated.png").resolve())
    assert calls == [(input_path, tmp_path / "page.translated.png")]
