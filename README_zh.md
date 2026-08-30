# ComicTranslate

[English](README.md) | [简体中文](README_zh.md)

ComicTranslate 是一款面向 Apple Silicon macOS 的本地漫画翻译工具。它将文字检测、OCR、简体中文翻译、原文擦除和译文排版串联成一条完整流水线，并尽可能保留输入图片的尺寸与透明通道。输入既可以是单张图片，也可以是一个文件夹第一层中的所有受支持图片。

> [!IMPORTANT]
> 当前版本处于早期开发阶段，仅支持 Apple Silicon macOS 和 Python 3.10.18。所有模型都必须提前下载到本地，程序不会自动下载模型。

## 目录

- [功能特性](#功能特性)
- [工作流程](#工作流程)
- [运行要求](#运行要求)
- [快速开始](#快速开始)
- [模型配置](#模型配置)
- [命令行用法](#命令行用法)
- [Python API](#python-api)
- [调试输出](#调试输出)
- [项目结构](#项目结构)
- [开发与测试](#开发与测试)
- [已知限制](#已知限制)
- [常见问题](#常见问题)
- [参与贡献](#参与贡献)
- [第三方代码与许可](#第三方代码与许可)

## 功能特性

- 使用 RT-DETR 检测气泡文字和独立文字区域。
- 使用 Qwen VLM 结合目标区域与整页上下文完成 OCR 和简体中文翻译。
- 自动跳过已有中文、装饰图案和无法辨认的误检区域。
- 使用 comic-text-detector ONNX 模型生成像素级文字掩膜。
- 使用 LaMa 擦除所有待翻译的原文字。
- 根据区域宽高比自动选择横排或竖排，并自动调整字号。
- 支持将译文放入气泡安全区，或回填到原文字检测框。
- 支持 PNG、JPEG 和 WebP，保持原始像素尺寸；PNG/WebP 的 alpha 通道会被保留。
- 支持批量翻译文件夹第一层中的图片，不递归处理子文件夹。
- 采用原子写入：关键阶段失败时不会写入不完整的新结果文件。
- 可导出检测、OCR、翻译、掩膜和修复图等调试产物。

## 工作流程

```text
输入图片
  -> 环境与模型校验
  -> RT-DETR 区域检测
  -> Qwen VLM OCR 与翻译
  -> ONNX 文字掩膜
  -> LaMa 原文擦除
  -> 中文横排/竖排渲染
  -> 原子保存输出
```

竖排文字在单列内自上而下书写，多列按从左到右排列。默认的 `bubble` 模式会在气泡边缘内缩后排版；`original` 模式则使用原文字检测框。若译文在最小字号下仍无法完全放入目标区域，程序会保持最小可读字号并允许居中的轻微越界，而不会中止整页处理。

## 运行要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Apple Silicon macOS（arm64） |
| Python | **3.10.18**，必须精确匹配 |
| 计算后端 | Apple Metal / MPS |
| 包管理器 | [uv](https://docs.astral.sh/uv/) |
| 模型 | RT-DETR、Qwen MLX、comic-text-detector ONNX、LaMa |
| 输入格式 | `.png`、`.jpg`、`.jpeg`、`.webp` |
| 输出格式 | `.png`、`.jpg`、`.jpeg`、`.webp` |

带 alpha 通道的输入不能输出为 JPEG，请选择 PNG 或 WebP。

## 快速开始

### 1. 安装依赖

克隆仓库并进入项目目录后，安装锁定的运行时和开发依赖：

```bash
uv sync --all-groups
```

### 2. 准备模型

提前将四个模型下载到本地，并记录各自的绝对路径。模型不会由 ComicTranslate 自动下载。模型类型与参数的对应关系见[模型配置](#模型配置)。

### 3. 翻译图片

```bash
uv run comictranslate page.png \
  --detector-model /absolute/path/to/comic-text-and-bubble-detector \
  --qwen-model /absolute/path/to/qwen-mlx-model \
  --text-mask-model /absolute/path/to/comictextdetector.pt.onnx \
  --lama-model /absolute/path/to/big-lama.pt
```

未指定 `--output` 时，结果写入输入图片旁的 `<原文件名>.translated.png`。例如，`page.jpg` 对应 `page.translated.png`。

翻译文件夹时，将文件夹作为 `INPUT`：

```bash
uv run comictranslate ./pages \
  --detector-model /absolute/path/to/comic-text-and-bubble-detector \
  --qwen-model /absolute/path/to/qwen-mlx-model \
  --text-mask-model /absolute/path/to/comictextdetector.pt.onnx \
  --lama-model /absolute/path/to/big-lama.pt
```

文件夹模式只扫描第一层，默认将结果写入 `./pages/translated/`。已有结果会直接跳过，因此中断后可以再次运行以继续处理。

## 模型配置

| 阶段 | 使用的模型 | 下载地址 | CLI 参数与路径类型 |
| --- | --- | --- | --- |
| 气泡与文字检测 | `ogkalu/comic-text-and-bubble-detector` | [Hugging Face（固定版本 `16e8a62`）](https://huggingface.co/ogkalu/comic-text-and-bubble-detector/tree/16e8a622f91fabc6b5b65c96d32d1183f8843546) | `--detector-model`（模型目录） |
| OCR 与翻译 | `mlx-community/Qwen3.5-9B-MLX-4bit` | [Hugging Face（固定版本 `938d891`）](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit/tree/938d8919941c6e7efd3c7150eff7fe9d12afa631) | `--qwen-model`（MLX 模型目录） |
| 精细文字掩膜 | `comictextdetector.pt.onnx` | [GitHub Release 直接下载](https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx) | `--text-mask-model`（ONNX 文件） |
| 图像修复 | `big-lama.pt`（TorchScript） | [Hugging Face 直接下载](https://huggingface.co/okaris/simple-lama/resolve/d5706085cdbdd5eb72503fdcd9fa648e952cfa53/big-lama.pt) | `--lama-model`（模型文件） |

当前代码中的开发环境默认路径为：

```text
/Volumes/yrq/models/
├── comic-text-and-bubble-detector/
├── Qwen/Qwen3.5-9B-Official-MLX-4bit/
├── manga-image-translator/comictextdetector.pt.onnx
└── lama/big-lama.pt
```

这些默认值是开发环境中的绝对路径。除非本机目录完全一致，否则必须通过对应的 CLI 参数或 `PipelineConfig` 显式覆盖。四个模型路径必须是绝对路径；CLI 支持以 `~/` 开头并自动展开。

下载后可以将模型保存到任意位置，本地目录名无需与表中的仓库名一致。为确认单文件模型与当前验证版本一致，可校验 SHA-256：

```text
comictextdetector.pt.onnx  1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f
big-lama.pt                7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c
```

默认情况下，程序会检查 `http://127.0.0.1:8080/v1`：

- 如果健康的 `mlx_vlm` 服务已存在，则直接复用且不会在退出时关闭它。
- 如果服务不存在，则使用当前 Python 环境启动 `mlx_vlm.server`，最多等待 180 秒，并仅在处理结束时关闭本次启动的进程。
- 如果端口被其他服务占用，则明确报错。可通过 `--server-url` 更换地址或端口。
- 远程 `mlx_vlm` 服务可以被复用，但程序不会自动启动远程服务。

字体未显式指定时，程序会依次尝试 macOS 自带的 STHeiti Medium、STHeiti Light 和 Arial Unicode。也可以使用 `--font` 指定其他字体文件。

## 命令行用法

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

| 参数 | 说明 |
| --- | --- |
| `INPUT` | 必填，待翻译的单张漫画图片或图片文件夹。 |
| `-o, --output` | 单图模式下是输出文件（默认同目录 `<stem>.translated.png`）；文件夹模式下是输出目录（默认 `INPUT/translated/`）。 |
| `--debug-dir` | 保存中间结果的目录；文件夹模式会为每个源文件创建独立子目录。 |
| `--detector-model` | RT-DETR 模型目录的绝对路径。 |
| `--qwen-model` | Qwen MLX 模型目录的绝对路径。 |
| `--text-mask-model` | comic-text-detector ONNX 文件的绝对路径。 |
| `--lama-model` | `big-lama.pt` 文件的绝对路径。 |
| `--server-url` | OpenAI 兼容的 `mlx_vlm` API 地址；默认 `http://127.0.0.1:8080/v1`。 |
| `--font` | 中文字体文件；默认尝试系统字体。 |
| `--text-placement` | `bubble` 使用气泡安全区；`original` 使用原文字框。 |

完整示例：

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

成功后，标准输出会返回 JSON 摘要：

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

文件夹输入只处理第一层的 PNG、JPEG 和 WebP 文件，并按文件名排序。每个结果命名为 `<stem>.translated.png`，默认输出目录为 `INPUT/translated/`；不支持的文件和子文件夹会被忽略，已有结果会跳过。若两个输入文件具有相同 stem、会映射到同一个输出文件，命令会在处理前报错。

文件夹批次会在所有图片尝试完成后输出一次 JSON 汇总：

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

单张图片失败后，批次会继续处理其余图片；只要存在失败项，命令退出码就是 `1`。单图模式禁止将输出路径设置为输入图片本身。输出目录不存在时会自动创建。

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

`translate_image` 一次处理一张图片并返回 `PipelineResult`。`translate_directory` 返回 `BatchResult`，其中包含成功的 `PipelineResult`、跳过的输入路径和 `BatchFailure` 失败记录。所有公开类型和函数均从 `comictranslate` 包根导出。

## 调试输出

设置 `--debug-dir DIR` 后会生成：

```text
DIR/
├── detections.json       # 原始检测、去重结果与区域坐标
├── roi/
│   └── region-*.png      # 每个待 OCR 区域的裁剪图
├── translations.json     # OCR、动作与简体中文译文
├── mask.png              # 合并后的全图文字掩膜
└── clean.png             # LaMa 擦除原文后的图片
```

每次运行会清理同一调试目录中的上述旧产物和 `roi/region-*.png`，请勿将需要保留的文件放在这些位置。

文件夹模式会为每个源文件建立类似 `DIR/page.jpg/` 的隔离目录，内部结构与上面相同，因此一张图片的调试清理不会影响其他图片。

## 项目结构

```text
ComicTranslate/
├── src/comictranslate/
│   ├── cli.py            # CLI 参数与入口
│   ├── config.py         # 流水线配置和默认值
│   ├── detection.py      # RT-DETR 检测
│   ├── qwen.py           # Qwen 服务管理、OCR 与翻译
│   ├── masking.py        # ONNX 文字掩膜与拼接
│   ├── inpainting.py     # LaMa 图像修复
│   ├── rendering.py      # 中文排版与绘制
│   └── pipeline.py       # 端到端流水线编排
├── tests/                # pytest 单元测试与集成测试
├── LICENSES/             # 第三方许可全文
├── main.py               # 开发环境入口
├── pyproject.toml        # 项目元数据与依赖
└── uv.lock               # 锁定依赖版本
```

## 开发与测试

运行不加载真实模型的快速单元测试：

```bash
uv run pytest
```

使用本地模型运行 RT-DETR、ONNX 和 LaMa 集成测试：

```bash
RUN_MODEL_INTEGRATION=1 uv run pytest tests/test_model_integration.py
```

可通过环境变量 `COMICTRANSLATE_REFERENCE_IMAGE` 覆盖集成测试使用的参考图片：

```bash
COMICTRANSLATE_REFERENCE_IMAGE=/absolute/path/to/page.jpg \
RUN_MODEL_INTEGRATION=1 \
uv run pytest tests/test_model_integration.py
```

构建 wheel 和源码包：

```bash
uv build
```

真实模型集成测试默认跳过，不能将只运行单元测试表述为模型推理已经验证。

## 已知限制

- 文件夹批处理只扫描第一层，不支持递归子文件夹、PDF/EPUB 或图形界面。
- 目标语言固定为简体中文。
- 运行环境固定为 Apple Silicon macOS 和 Python 3.10.18。
- 模型不会自动下载，且默认模型路径具有开发机特定性。
- 版面检测、OCR、翻译、擦除和排版质量取决于本地模型与原图质量。
- 极小或不规则区域可能在最小字号下出现轻微文字越界。

## 常见问题

### 提示“缺少本地模型”

确认四个路径分别指向两个模型目录和两个模型文件，并使用绝对路径。最常见原因是仍在使用项目内置的开发机默认路径。

### 提示“Apple Metal 环境不可用”

确认系统为 Apple Silicon macOS、Python 版本精确为 3.10.18，且当前环境中的 MLX 可以访问 Metal。

### 提示端口被非 `mlx_vlm` 服务占用

停止占用对应端口的程序，或通过 `--server-url http://127.0.0.1:其他端口/v1` 选择空闲端口。

### 带透明通道的图片无法保存为 JPEG

JPEG 不支持 alpha 通道。将输出扩展名改为 `.png` 或 `.webp`。

### 找不到中文字体

使用 `--font /absolute/path/to/font.ttf` 指定支持简体中文的 TrueType/OpenType 字体文件。

## 参与贡献

欢迎通过 Issue 或 Pull Request 改进项目。提交前请：

1. 保持改动聚焦，不提交本地模型、翻译结果、调试目录或机器特定路径。
2. 为行为变更补充对应的 pytest 测试。
3. 运行 `uv run pytest`，并明确说明是否运行了模型集成测试。
4. 使用 Conventional Commits，例如 `fix(rendering): 修正竖排文字对齐`。

涉及渲染或流水线输出的变更，建议在 Pull Request 中附上处理前后图片。

## 第三方代码与许可

项目包含经裁剪和修改的 [`dmMaze/comic-text-detector`](https://github.com/dmMaze/comic-text-detector) 纯推理 mask refinement 代码。其固定提交、修改范围和许可信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，GPL-3.0 许可全文见 [LICENSES/comic-text-detector-GPL-3.0.txt](LICENSES/comic-text-detector-GPL-3.0.txt)。

分发包含该代码的组合程序时，需要遵守 GPL-3.0 的相应要求。模型文件不随本仓库分发，其许可由各模型提供方单独规定。
