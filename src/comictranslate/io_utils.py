from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .errors import ConfigurationError


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def default_output_path(input_path: Path) -> Path:
    input_path = Path(input_path)
    return input_path.with_name(f"{input_path.stem}.translated.png")


def default_output_directory(input_dir: Path) -> Path:
    return Path(input_dir) / "translated"


def discover_images(input_dir: Path) -> tuple[Path, ...]:
    input_dir = Path(input_dir).expanduser()
    if not input_dir.is_dir():
        raise ConfigurationError(f"输入文件夹不存在: {input_dir}")
    images = tuple(
        sorted(
            (
                path
                for path in input_dir.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )
    if not images:
        raise ConfigurationError(f"输入文件夹中没有支持的图片: {input_dir}")
    return images


def directory_output_path(input_path: Path, output_dir: Path) -> Path:
    return Path(output_dir) / default_output_path(input_path).name


def validate_io_paths(input_path: Path, output_path: Path) -> tuple[Path, Path]:
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    if not input_path.is_file():
        raise ConfigurationError(f"输入图片不存在: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ConfigurationError("输入仅支持 PNG、JPEG 和 WebP")
    if output_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ConfigurationError("输出仅支持 PNG、JPEG 和 WebP")
    if input_path.resolve() == output_path.resolve(strict=False):
        raise ConfigurationError("禁止覆盖输入图片")
    return input_path, output_path


def load_image(input_path: Path) -> Image.Image:
    try:
        with Image.open(input_path) as opened:
            opened.load()
            return opened.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ConfigurationError(f"无法读取图片 {input_path}: {exc}") from exc


def original_alpha(image: Image.Image) -> Image.Image | None:
    return image.getchannel("A").copy() if "A" in image.getbands() else None


def restore_alpha(image: Image.Image, alpha: Image.Image | None) -> Image.Image:
    if alpha is None:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def atomic_save(image: Image.Image, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"} and "A" in image.getbands():
        raise ConfigurationError("带 alpha 通道的输入不能输出为 JPEG，请使用 PNG 或 WebP")
    image_format = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}[suffix]
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        image.save(temporary_path, format=image_format)
        with temporary_path.open("rb") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary_path, output_path)
    except OSError as exc:
        raise ConfigurationError(f"无法写入输出图片 {output_path}: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
