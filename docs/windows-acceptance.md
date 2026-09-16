# Windows / macOS acceptance record

Status: implementation and automated regression only; **hardware parity is not certified**.

| Target | PyTorch | Real GPU/model acceptance |
| --- | --- | --- |
| Apple Silicon macOS | MPS | Pending current-change model regression |
| Windows 10 22H2 / 11, RTX 20/30/40 | CUDA 12.6 default | Pending |
| Windows 10 22H2 / 11, RTX 50 | CUDA 13 optional | Pending |

Use llama.cpp b10981 and Qwen3.5-9B Q4_K_M with its matching projector. For each machine record OS/build, GPU/VRAM, RAM, driver, Torch/TorchVision versions, llama runtime CUDA build/version, model repository/revision, SHA256 of runtime/model/projector, elapsed time and peak RAM/VRAM. Do not invent minimum memory requirements before measuring.

Set `RUN_MODEL_INTEGRATION=1`, `COMICTRANSLATE_DEVICE=cuda` (or `mps`), `COMICTRANSLATE_REFERENCE_IMAGE`, `COMICTRANSLATE_DETECTOR_MODEL`, `COMICTRANSLATE_TEXT_MASK_MODEL`, `COMICTRANSLATE_LAMA_MODEL`, `COMICTRANSLATE_QWEN_MODEL`; for llama also set `COMICTRANSLATE_QWEN_MODE=managed-llama`, `COMICTRANSLATE_QWEN_SERVER` and `COMICTRANSLATE_QWEN_MMPROJ`. Run `uv run --no-sync pytest tests/test_model_integration.py`.

Manually verify PNG/JPEG/WebP, alpha, Chinese skip, horizontal/vertical text, bubble/original placement, Chinese and spaced paths (including fonts and models), batch ordering/skips/failure continuation, debug outputs, occupied output files, startup/inference cancellation and process cleanup. Verify real multi-image OCR IDs and translation meaning visually. Compare fixed-font mocked translations across systems for dimensions/layout, without requiring identical real-model wording or pixels.

Release gate: both Windows versions must translate one image and a directory in one command after model preparation, clean up owned servers, and pass macOS regression. Save actual observations and hashes here only after the runs. Native CI must also pass; merely adding a workflow does not count as a successful run.

## Automated evidence (2026-09-16)

- Apple Silicon local regression: `104 passed, 4 skipped`; skipped tests require real model inference.
- CLI `--help` smoke check and `git diff --check`: passed.
- Windows `uv sync --locked --dry-run --python-platform x86_64-pc-windows-msvc --all-groups`: selects Torch 2.13.0+cu126 / TorchVision 0.28.0+cu126.
- Same dry run with `--no-group cuda126 --extra cuda13`: selects Torch 2.13.0+cu130 / TorchVision 0.28.0+cu130.
- Dry runs do not download/install Windows packages. Native CI and real-model acceptance are pending.

Protocol reference: [llama.cpp b10981 server](https://github.com/ggml-org/llama.cpp/blob/b10981/tools/server/README.md). Reuse cannot verify the precise mmproj filename because this version's properties endpoint exposes vision capability but not projector provenance.
