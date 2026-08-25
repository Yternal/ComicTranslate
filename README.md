# ComicTranslate

[English](README.md) | [简体中文](README_zh.md)

ComicTranslate is a local, single-image comic translation tool for Apple Silicon Macs. It combines text detection, OCR, Simplified Chinese translation, source-text removal, and translated-text rendering in one pipeline while preserving the input dimensions and transparency whenever possible.

> [!IMPORTANT]
> This project is in an early stage. The current version supports only Apple Silicon macOS, Python 3.10.18, and one image at a time. All models must be downloaded in advance; ComicTranslate never downloads models automatically.

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
| Operating system | Apple Silicon macOS (arm64) |
| Python | **3.10.18**, exact version required |
| Compute backend | Apple Metal / MPS |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Models | RT-DETR, Qwen MLX, comic-text-detector ONNX, and LaMa |
| Input formats | `.png`, `.jpg`, `.jpeg`, `.webp` |
| Output formats | `.png`, `.jpg`, `.jpeg`, `.webp` |

An input with an alpha channel cannot be written as JPEG. Use PNG or WebP instead.

## Quick start

### 1. Install dependencies

Clone the repository, enter its directory, and install the locked runtime and development dependencies:

```bash
uv sync --all-groups
```

### 2. Prepare the models

Download all four models in advance and note their absolute paths. ComicTranslate does not download them for you. See [Model configuration](#model-configuration) for the mapping between models and arguments.

### 3. Translate an image

```bash
uv run comictranslate page.png \
  --detector-model /absolute/path/to/comic-text-and-bubble-detector \
  --qwen-model /absolute/path/to/qwen-mlx-model \
  --text-mask-model /absolute/path/to/comictextdetector.pt.onnx \
  --lama-model /absolute/path/to/big-lama.pt
```

Without `--output`, the result is created next to the input as `<original-name>.translated.png`. For example, `page.jpg` produces `page.translated.png`.

## Model configuration

| Stage | Model in use | Download | CLI argument and path type |
| --- | --- | --- | --- |
| Bubble and text detection | `ogkalu/comic-text-and-bubble-detector` | [Hugging Face (pinned revision `16e8a62`)](https://huggingface.co/ogkalu/comic-text-and-bubble-detector/tree/16e8a622f91fabc6b5b65c96d32d1183f8843546) | `--detector-model` (model directory) |
| OCR and translation | `mlx-community/Qwen3.5-9B-MLX-4bit` | [Hugging Face (pinned revision `938d891`)](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit/tree/938d8919941c6e7efd3c7150eff7fe9d12afa631) | `--qwen-model` (MLX model directory) |
| Refined text mask | `comictextdetector.pt.onnx` | [Direct GitHub Release download](https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx) | `--text-mask-model` (ONNX file) |
| Image inpainting | `big-lama.pt` (TorchScript) | [Direct Hugging Face download](https://huggingface.co/okaris/simple-lama/resolve/d5706085cdbdd5eb72503fdcd9fa648e952cfa53/big-lama.pt) | `--lama-model` (model file) |

The development defaults currently hard-coded in the project are:

```text
/Volumes/yrq/models/
├── comic-text-and-bubble-detector/
├── Qwen/Qwen3.5-9B-Official-MLX-4bit/
├── manga-image-translator/comictextdetector.pt.onnx
└── lama/big-lama.pt
```

These defaults are absolute paths from the development machine. Unless your local layout matches exactly, override them with CLI arguments or `PipelineConfig`. All four model paths must be absolute. The CLI expands paths beginning with `~/`.

Downloaded models can be stored anywhere, and local directory names do not need to match the repository names in the table. To verify that the single-file models match the versions tested here, compare their SHA-256 checksums:

```text
comictextdetector.pt.onnx  1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f
big-lama.pt                7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c
```

By default, ComicTranslate checks `http://127.0.0.1:8080/v1`:

- A healthy existing `mlx_vlm` service is reused and is not stopped on exit.
- If no service is available, ComicTranslate starts `mlx_vlm.server` with the active Python environment, waits for up to 180 seconds, and stops only the process it created.
- If another service owns the port, the command fails with an explicit error. Use `--server-url` to select a different address or port.
- An available remote `mlx_vlm` service can be reused, but ComicTranslate will not start one remotely.

When no font is specified, ComicTranslate tries the macOS STHeiti Medium, STHeiti Light, and Arial Unicode fonts in that order. Use `--font` to provide another font file.

## Command-line usage

```text
comictranslate INPUT
  [-o OUTPUT]
  [--debug-dir DIR]
  [--detector-model DIR]
  [--qwen-model DIR]
  [--text-mask-model FILE]
  [--lama-model FILE]
  [--server-url URL]
  [--font FILE]
  [--text-placement {bubble,original}]
```

| Argument | Description |
| --- | --- |
| `INPUT` | Required path to one comic image. |
| `-o, --output` | Output path; defaults to `<name>.translated.png`. |
| `--debug-dir` | Directory for intermediate artifacts. |
| `--detector-model` | Absolute path to the RT-DETR model directory. |
| `--qwen-model` | Absolute path to the Qwen MLX model directory. |
| `--text-mask-model` | Absolute path to the comic-text-detector ONNX file. |
| `--lama-model` | Absolute path to the `big-lama.pt` file. |
| `--server-url` | OpenAI-compatible `mlx_vlm` API URL; defaults to `http://127.0.0.1:8080/v1`. |
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

The output path must not point to the input image. Missing output directories are created automatically.

## Python API

```python
from pathlib import Path

from comictranslate import PipelineConfig, translate_image

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
```

`translate_image` processes one image and returns a `PipelineResult`. The public API is exported from the `comictranslate` package root.

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

- Processes one image at a time; there is no directory batch mode, PDF/EPUB support, or GUI.
- The target language is fixed to Simplified Chinese.
- The runtime is restricted to Apple Silicon macOS and Python 3.10.18.
- Models are never downloaded automatically, and the built-in model paths are development-machine specific.
- Detection, OCR, translation, removal, and layout quality depend on the local models and source image.
- Very small or irregular regions may show slight text overflow at the minimum font size.

## Troubleshooting

### “Missing local models”

Confirm that the four paths point to two model directories and two model files, all with absolute paths. The most common cause is unintentionally using the development-machine defaults.

### “Apple Metal environment unavailable”

Confirm that the machine is an Apple Silicon Mac, Python is exactly 3.10.18, and MLX can access Metal in the active environment.

### The port is occupied by a non-`mlx_vlm` service

Stop the process using that port, or pass `--server-url http://127.0.0.1:ANOTHER_PORT/v1` with an available port.

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
