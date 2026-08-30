from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from .config import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_LAMA_MODEL,
    DEFAULT_QWEN_MODEL,
    DEFAULT_SERVER_URL,
    DEFAULT_TEXT_MASK_MODEL,
    PipelineConfig,
)
from .errors import ComicTranslateError
from .io_utils import default_output_path
from .pipeline import translate_directory, translate_image


def absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"必须使用绝对路径: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comictranslate", description="将漫画图片中的外文翻译并回填为简体中文"
    )
    parser.add_argument(
        "input", type=Path, metavar="INPUT", help="单张漫画图片或图片文件夹"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="单图输出文件或批量输出文件夹"
    )
    parser.add_argument("--debug-dir", type=Path, help="保存中间调试产物的目录")
    parser.add_argument(
        "--detector-model",
        type=absolute_path,
        default=DEFAULT_DETECTOR_MODEL,
        help="RT-DETR 模型目录绝对路径",
    )
    parser.add_argument(
        "--qwen-model",
        type=absolute_path,
        default=DEFAULT_QWEN_MODEL,
        help="Qwen MLX 模型目录绝对路径",
    )
    parser.add_argument(
        "--text-mask-model",
        type=absolute_path,
        default=DEFAULT_TEXT_MASK_MODEL,
        help="comic-text-detector ONNX 文件绝对路径",
    )
    parser.add_argument(
        "--lama-model",
        type=absolute_path,
        default=DEFAULT_LAMA_MODEL,
        help="big-lama.pt 文件绝对路径",
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--font", type=Path)
    parser.add_argument(
        "--text-placement",
        choices=("bubble", "original"),
        default="bubble",
        help="使用 bubble 气泡安全区，或回填到 original 原文字框",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = PipelineConfig(
            detector_model=args.detector_model,
            qwen_model=args.qwen_model,
            text_mask_model=args.text_mask_model,
            lama_model=args.lama_model,
            server_url=args.server_url,
            font_path=args.font,
            debug_dir=args.debug_dir,
            text_placement=args.text_placement,
        )
        if args.input.expanduser().is_dir():
            result = translate_directory(args.input, args.output, config)
            exit_code = 1 if result.failed else 0
        else:
            output = args.output or default_output_path(args.input)
            result = translate_image(args.input, output, config)
            exit_code = 0
    except (ComicTranslateError, ValueError) as exc:
        logger.error("处理失败：{}", exc)
        return 1
    except Exception as exc:
        logger.exception("处理失败：发生未预期错误：{}", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("已取消")
        return 130
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
