# ComicTranslate

[English](README.md) | [简体中文](README_zh.md)

ComicTranslate is a local comic translation tool for Apple Silicon macOS and Windows x64 with NVIDIA GPUs. It combines text detection, OCR, Simplified Chinese translation, source-text removal, and translated-text rendering in one pipeline while preserving the input dimensions and transparency whenever possible. It accepts either one image or all supported images directly inside a folder.

> [!IMPORTANT]
> This project is in an early stage. The current version supports Apple Silicon macOS and Windows x64, and requires Python 3.10.18 exactly. Models, fonts, and external Qwen services are never downloaded or installed automatically.

## Table of contents

- [Features](#features)
  - [How it works](#how-it-works)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Model configuration](#model-configuration)
- [Command-line usage](#command-line-usage)
- [Python API](#python-api)
- [Debug output](#debug-output)
- [Project structure](#project-structure)
- [Development and testing](#development-and-testing)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Third-party code and licensing](#third-party-code-and-licensing)

## Features

- Detects speech-bubble and free-standing text regions with RT-DETR.
- Uses a Qwen VLM to perform OCR and translate into Simplified Chinese with both ROI and full-page context.
- Skips text that is already Chinese, decorative elements, and unreadable false positives.
- Produces pixel-level text masks with the comic-text-detector ONNX model.
- Removes all source text selected for translation with LaMa.
- Selects horizontal or vertical layout from the target region and adjusts the font size automatically.
- Places translated text in a conservative speech-bubble safe area or in the original detected text box.
- Reads and writes PNG, JPEG, and WebP; preserves the original pixel dimensions and PNG/WebP alpha channel.
- Translates the supported images directly inside a folder without descending into subfolders.
- Saves atomically so a failed critical stage does not produce a partial new result file.
- Exports optional detection, OCR, translation, mask, and inpainting artifacts for debugging.

## How it works

```text
Input image
  -> Validate runtime and models
  -> Detect regions with RT-DETR
  -> OCR and translate with Qwen VLM
  -> Build an ONNX text mask
  -> Remove source text with LaMa
  -> Render horizontal/vertical Chinese text
  -> Save the output atomically
```

Vertical text runs from top to bottom inside each column, with columns ordered from left to right. The default `bubble` mode lays text out inside an inset speech-bubble safe area; `original` uses the original text detection box. If a translation still does not fit at the minimum font size, ComicTranslate keeps that readable minimum and permits slight centered overflow instead of failing the whole page.

## Requirements

| Item | Requirement |
| --- | --- |
| Operating system | Apple Silicon macOS (arm64), or Windows x64 (AMD64) |
| Python | **3.10.18**, exact version required |
| Compute backend | macOS: MPS; Windows: NVIDIA CUDA 12.6, with CPU fallback in `auto` mode |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Models | RT-DETR, comic-text-detector ONNX, LaMa, plus Qwen MLX on macOS or a local OpenAI-compatible VLM service on Windows |
| Input formats | `.png`, `.jpg`, `.jpeg`, `.webp` |
| Output formats | `.png`, `.jpg`, `.jpeg`, `.webp` |

An input with an alpha channel cannot be written as JPEG. Use PNG or WebP instead.

## Quick start

### 1. Install dependencies

Clone the repository, enter its directory, and install the locked runtime and development dependencies:

```bash
uv sync --all-groups
```

On Windows, the lock file selects official PyTorch CUDA 12.6 wheels. Verify the result in PowerShell:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The second line should be `True` on the supported Windows target. A recent NVIDIA driver compatible with the bundled CUDA 12.6 runtime is required.

### 2. Prepare models and Qwen

Download RT-DETR, comic-text-detector ONNX, and LaMa in advance and note their absolute paths. On macOS, also download the Qwen MLX model. On Windows, start a local OpenAI-compatible vision-language-model service and note its model ID. See [Model configuration](#model-configuration) for details.

### 3. Translate an image

```bash
uv run comictranslate page.png \
  --detector-model /absolute/path/to/comic-text-and-bubble-detector \
  --qwen-model /absolute/path/to/qwen-mlx-model \
  --text-mask-model /absolute/path/to/comictextdetector.pt.onnx \
  --lama-model /absolute/path/to/big-lama.pt
```

Windows PowerShell example:

```powershell
uv run comictranslate "C:\Comics\page.png" `
  --output "C:\Comics\output\page.zh-CN.png" `
  --detector-model "C:\Models\comic-detector" `
  --text-mask-model "C:\Models\comictextdetector.pt.onnx" `
  --lama-model "C:\Models\big-lama.pt" `
  --device cuda `
  --qwen-service-mode external `
  --server-url http://127.0.0.1:8000/v1 `
  --qwen-model-id qwen-vl-local
```

Without `--output`, the result is created next to the input as `<original-name>.translated.png`. For example, `page.jpg` produces `page.translated.png`.

To translate a folder, pass the folder as `INPUT`:

```bash
uv run comictranslate ./pages \
  --detector-model /absolute/path/to/comic-text-and-bubble-detector \
  --qwen-model /absolute/path/to/qwen-mlx-model \
  --text-mask-model /absolute/path/to/comictextdetector.pt.onnx \
  --lama-model /absolute/path/to/big-lama.pt
```

The folder mode scans only the first level and writes results to `./pages/translated/` by default. Existing result files are skipped so an interrupted batch can be resumed.

## Model configuration

| Stage | Model in use | Download | CLI argument and path type |
| --- | --- | --- | --- |
| Bubble and text detection | `ogkalu/comic-text-and-bubble-detector` | [Hugging Face (pinned revision `16e8a62`)](https://huggingface.co/ogkalu/comic-text-and-bubble-detector/tree/16e8a622f91fabc6b5b65c96d32d1183f8843546) | `--detector-model` (model directory) |
| OCR and translation on macOS | `mlx-community/Qwen3.5-9B-MLX-4bit` | [Hugging Face (pinned revision `938d891`)](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit/tree/938d8919941c6e7efd3c7150eff7fe9d12afa631) | `--qwen-model` (MLX model directory) |
| Refined text mask | `comictextdetector.pt.onnx` | [Direct GitHub Release download](https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx) | `--text-mask-model` (ONNX file) |
| Image inpainting | `big-lama.pt` (TorchScript) | [Direct Hugging Face download](https://huggingface.co/okaris/simple-lama/resolve/d5706085cdbdd5eb72503fdcd9fa648e952cfa53/big-lama.pt) | `--lama-model` (model file) |

There are no built-in model paths. Required model arguments must be absolute. The CLI expands paths beginning with `~/` on macOS and accepts drive-qualified paths such as `C:\Models\big-lama.pt` on Windows.

Downloaded models can be stored anywhere, and local directory names do not need to match the repository names in the table. To verify that the single-file models match the versions tested here, compare their SHA-256 checksums:

```text
comictextdetector.pt.onnx  1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f
big-lama.pt                7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c
```

Qwen service mode defaults depend on the platform:

- `auto` resolves to `managed-mlx` on Apple Silicon macOS. It reuses a healthy `mlx_vlm` service or starts one with `--qwen-model`, and stops only the process it created.
- `auto` resolves to `external` on Windows. `external` accepts only `localhost`, `127.0.0.1`, or `::1`, calls `GET /v1/models`, and requires `--qwen-model-id` to match an advertised ID. It never starts or stops a process.
- The external service must implement `POST /v1/chat/completions`, OpenAI-style Data URL image content, and `response_format.type=json_schema`, without authentication. ComicTranslate sends the model ID and omits MLX-only fields.

On macOS, ComicTranslate tries STHeiti Medium, STHeiti Light, and Arial Unicode. On Windows, it checks `%WINDIR%\Fonts` for Microsoft YaHei, SimHei, and SimSun in that order. Use `--font` when no suitable system font exists.

## Command-line usage

```text
comictranslate INPUT
  [-o OUTPUT]
  [--debug-dir DIR]
  [--detector-model DIR]
  [--qwen-model DIR]
  [--text-mask-model FILE]
  [--lama-model FILE]
  [--device {auto,cpu,cuda,mps}]
  [--qwen-service-mode {auto,managed-mlx,external}]
  [--qwen-model-id MODEL_ID]
  [--server-url URL]
  [--font FILE]
  [--text-placement {bubble,original}]
```

| Argument | Description |
| --- | --- |
| `INPUT` | Required path to one comic image or a folder of images. |
| `-o, --output` | Output file for one image (default: sibling `<stem>.translated.png`), or output directory for folder mode (default: `INPUT/translated/`). |
| `--debug-dir` | Directory for intermediate artifacts; folder mode creates one subdirectory per source file. |
| `--detector-model` | Absolute path to the RT-DETR model directory. |
| `--qwen-model` | Absolute path to the Qwen MLX model directory. |
| `--text-mask-model` | Absolute path to the comic-text-detector ONNX file. |
| `--lama-model` | Absolute path to the `big-lama.pt` file. |
| `--device` | Shared RT-DETR/LaMa device. `auto` chooses CUDA, then MPS, then CPU. An unavailable explicit device is an error. |
| `--qwen-service-mode` | `auto`, macOS-only `managed-mlx`, or loopback-only `external`. |
| `--qwen-model-id` | Model ID advertised by an external service; required in `external` mode. |
| `--server-url` | OpenAI-compatible API base URL; defaults to `http://127.0.0.1:8080/v1`. |
| `--font` | Chinese font file; system fonts are tried by default. |
| `--text-placement` | `bubble` uses the bubble safe area; `original` uses the original text box. |

Complete example:

```bash
uv run comictranslate ./examples/page.webp \
  --output ./output/page.zh-CN.png \
  --debug-dir ./output/debug \
  --detector-model /models/comic-text-and-bubble-detector \
  --qwen-model /models/Qwen3.5-9B-MLX-4bit \
  --text-mask-model /models/comictextdetector.pt.onnx \
  --lama-model /models/big-lama.pt \
  --server-url http://127.0.0.1:8080/v1 \
  --text-placement bubble
```

On success, the command writes a JSON summary to standard output:

```json
{
  "width": 1057,
  "height": 1500,
  "translated_regions": 8,
  "skipped_regions": 1,
  "output_path": "/absolute/path/page.zh-CN.png",
  "region_ids": ["region-0001", "region-0002"]
}
```

For folder input, only first-level PNG, JPEG, and WebP files are processed, in filename order. Each result is named `<stem>.translated.png`; the default output directory is `INPUT/translated/`. Unsupported files and subdirectories are ignored. Existing results are skipped. If two inputs have the same stem and would map to one result, the command fails before processing anything.

A folder batch writes one JSON summary after all images have been attempted:

```json
{
  "input_dir": "/absolute/path/pages",
  "output_dir": "/absolute/path/pages/translated",
  "total": 3,
  "succeeded": 1,
  "skipped": 1,
  "failed": 1,
  "results": [
    {
      "width": 1057,
      "height": 1500,
      "translated_regions": 8,
      "skipped_regions": 1,
      "output_path": "/absolute/path/pages/translated/001.translated.png",
      "region_ids": ["region-0001"]
    }
  ],
  "skipped_inputs": ["/absolute/path/pages/002.jpg"],
  "failures": [
    {
      "input_path": "/absolute/path/pages/003.webp",
      "output_path": "/absolute/path/pages/translated/003.translated.png",
      "error_type": "TranslationError",
      "message": "..."
    }
  ]
}
```

The batch continues after an individual image fails and returns exit code `1` if any failures were recorded. The output path for one image must not point to the input itself. Missing output directories are created automatically.

## Python API

```python
from pathlib import Path

from comictranslate import PipelineConfig, translate_directory, translate_image

config = PipelineConfig(
    detector_model=Path("/models/comic-text-and-bubble-detector"),
    qwen_model=Path("/models/Qwen3.5-9B-MLX-4bit"),
    text_mask_model=Path("/models/comictextdetector.pt.onnx"),
    lama_model=Path("/models/big-lama.pt"),
    debug_dir=Path("debug"),
    text_placement="original",
)

result = translate_image("page.webp", "page.translated.png", config)
print(result.to_dict())

batch = translate_directory("pages", config=config)
print(batch.succeeded, batch.skipped, batch.failed)
```

On Windows, replace `qwen_model` with `qwen_service_mode="external"`, `qwen_model_id="qwen-vl-local"`, and the local service URL. Set `device="cuda"` to require CUDA, or leave `device="auto"` to warn and fall back to CPU when CUDA is unavailable.

`translate_image` processes one image and returns a `PipelineResult`. `translate_directory` returns a `BatchResult` containing successful `PipelineResult` objects, skipped input paths, and `BatchFailure` records. All public API types and functions are exported from the `comictranslate` package root.

## Debug output

Passing `--debug-dir DIR` creates:

```text
DIR/
├── detections.json       # Raw/normalized detections and region coordinates
├── roi/
│   └── region-*.png      # Crop for each OCR target
├── translations.json     # OCR text, action, and Simplified Chinese translation
├── mask.png              # Stitched full-page text mask
└── clean.png             # Image after LaMa source-text removal
```

Each run removes old copies of these artifacts and `roi/region-*.png` from the same debug directory. Do not store files you need to keep at those locations.

In folder mode, each source gets an isolated directory such as `DIR/page.jpg/` with the same layout. Debug data from one image therefore cannot clear another image's artifacts.

## Project structure

```text
ComicTranslate/
├── src/comictranslate/
│   ├── cli.py            # CLI arguments and entry point
│   ├── config.py         # Pipeline configuration and defaults
│   ├── detection.py      # RT-DETR detection
│   ├── qwen.py           # Qwen service, OCR, and translation
│   ├── masking.py        # ONNX text masks and stitching
│   ├── inpainting.py     # LaMa image inpainting
│   ├── rendering.py      # Chinese text layout and drawing
│   └── pipeline.py       # End-to-end orchestration
├── tests/                # pytest unit and integration tests
├── LICENSES/             # Full third-party license texts
├── main.py               # Development entry point
├── pyproject.toml        # Project metadata and dependencies
└── uv.lock               # Locked dependency versions
```

## Development and testing

Run the fast unit suite without loading real models:

```bash
uv run pytest
```

Run the RT-DETR, ONNX, and LaMa integration tests with local model files:

```bash
RUN_MODEL_INTEGRATION=1 uv run pytest tests/test_model_integration.py
```

Set `COMICTRANSLATE_DETECTOR_MODEL`, `COMICTRANSLATE_TEXT_MASK_MODEL`, and `COMICTRANSLATE_LAMA_MODEL` to absolute paths before running the corresponding integration checks.

Override the reference image used by the integration tests with `COMICTRANSLATE_REFERENCE_IMAGE`:

```bash
COMICTRANSLATE_REFERENCE_IMAGE=/absolute/path/to/page.jpg \
RUN_MODEL_INTEGRATION=1 \
uv run pytest tests/test_model_integration.py
```

Build the wheel and source distribution:

```bash
uv build
```

Real-model integration tests are skipped by default. A unit-test-only run must not be reported as validated model inference.

## Known limitations

- Folder batches scan one level only; recursive folders, PDF/EPUB input, and a GUI are not supported.
- The target language is fixed to Simplified Chinese.
- Supported runtimes are Apple Silicon macOS and Windows x64 with Python 3.10.18. Windows 10, Windows ARM64, Linux, AMD/Intel GPUs, remote services, and authenticated services are not supported in the first Windows release.
- Models, fonts, and Qwen services are never downloaded or installed automatically.
- CPU fallback preserves the functional path but is not expected to provide practical model-inference performance.
- Detection, OCR, translation, removal, and layout quality depend on the local models and source image.
- Very small or irregular regions may show slight text overflow at the minimum font size.

## Troubleshooting

### “Missing local models”

Confirm that RT-DETR points to a directory and the text-mask and LaMa paths point to files. `managed-mlx` additionally requires a Qwen model directory; `external` requires a model ID instead. All local model paths must be absolute.

### CUDA is unavailable on Windows

Run the PowerShell verification command above. Confirm that the NVIDIA driver is current and that `uv.lock` installed a `+cu126` Torch build. `--device cuda` fails immediately when CUDA is unavailable; `--device auto` warns and uses CPU.

### The Qwen service cannot be used

For `managed-mlx`, stop the conflicting process or choose another port. For `external`, confirm the URL is loopback-only, `GET /v1/models` contains the exact configured model ID, and the server supports Data URL vision input plus strict JSON Schema responses. ComicTranslate does not stop an external service when it exits.

### An image with transparency cannot be saved as JPEG

JPEG does not support alpha channels. Change the output extension to `.png` or `.webp`.

### No Chinese font can be found

Pass `--font /absolute/path/to/font.ttf` with a TrueType/OpenType font that supports Simplified Chinese.

## Contributing

Issues and pull requests are welcome. Before submitting a change:

1. Keep the change focused. Do not commit local models, translated images, debug output, or machine-specific paths.
2. Add focused pytest coverage for behavior changes.
3. Run `uv run pytest` and state clearly whether model integration tests were run.
4. Use Conventional Commits, for example `fix(rendering): correct vertical text alignment`.

For rendering or pipeline-output changes, include before-and-after images in the pull request when possible.

## Third-party code and licensing

This project contains a trimmed and modified inference-only mask-refinement adaptation of [`dmMaze/comic-text-detector`](https://github.com/dmMaze/comic-text-detector). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the pinned commit, modifications, and licensing information, and [LICENSES/comic-text-detector-GPL-3.0.txt](LICENSES/comic-text-detector-GPL-3.0.txt) for the full GPL-3.0 text.

Distribution of a combined work containing that code must comply with the applicable GPL-3.0 requirements. Model files are not distributed with this repository and remain subject to their providers' separate license terms.
