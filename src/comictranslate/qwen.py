from __future__ import annotations

import base64
import io
import json
import math
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from .errors import TranslationError
from .models import Region, Translation


TRANSLATION_PROMPT = """你是专业的漫画 OCR 与翻译器。前面的图片是目标 ROI，每张顶部白条中的 TARGET ID 是该图片的唯一 ID；最后一张图是整页上下文。
结合整页的画面、人物关系、前后对话与阅读顺序，逐个处理指定 ID：
- source_text 必须只识别对应 TARGET ID 的 ROI 正文；整页图片仅用于理解语境，禁止从整页中另选文字或按整页阅读顺序重新分配 ID。
- 可读外文、拟声词、感叹词、旁白和独立文字：action=translate，识别原文写入 source_text，并自然翻译为简体中文写入 translation。
- 已经是中文、纯装饰图案或无法辨认的误检：action=skip，不得猜测，translation 必须为空字符串。
- 不增加原文没有的信息；被竖排、换行拆开的文字按语义合并；拟声词使用自然的中文漫画表达。
- 每个给定 ID 必须且只能返回一次，禁止返回其他 ID。只返回符合 JSON Schema 的内容。"""


TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "source_text", "action", "translation"],
                "properties": {
                    "id": {"type": "string"},
                    "source_text": {"type": "string"},
                    "action": {"type": "string", "enum": ["translate", "skip"]},
                    "translation": {"type": "string"},
                },
            },
        }
    },
}


def _root_url(server_url: str) -> str:
    parsed = urlsplit(server_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def _api_url(server_url: str, endpoint: str) -> str:
    return f"{server_url.rstrip('/')}/{endpoint.lstrip('/')}"


def _open(request: urllib.request.Request, timeout: float):
    """Bypass macOS system proxies for loopback MLX traffic."""
    hostname = urlsplit(request.full_url).hostname
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=timeout
        )
    return urllib.request.urlopen(request, timeout=timeout)


class QwenServiceManager:
    def __init__(
        self,
        server_url: str,
        model_path: Path,
        *,
        start_timeout: float = 180.0,
        python_executable: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.model_path = Path(model_path)
        self.start_timeout = start_timeout
        self.python_executable = python_executable or sys.executable
        self._clock = clock
        self._sleep = sleeper
        self._process: subprocess.Popen[bytes] | None = None
        self._owned = False

    @property
    def owned(self) -> bool:
        return self._owned

    @property
    def health_url(self) -> str:
        return f"{_root_url(self.server_url)}/health"

    def _probe(self) -> Literal["mlx", "other", "unreachable"]:
        request = urllib.request.Request(
            self.health_url,
            headers={"Accept": "application/json", "User-Agent": "comictranslate/0.1"},
        )
        try:
            with _open(request, timeout=2.0) as response:
                server_header = response.headers.get("Server", "")
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return "unreachable"
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "other"
        is_mlx = (
            isinstance(payload, dict)
            and payload.get("status") == "healthy"
            and (
                server_header.lower().startswith("mlx_vlm/")
                or "continuous_batching_enabled" in payload
            )
        )
        return "mlx" if is_mlx else "other"

    def _address(self) -> tuple[str, int]:
        parsed = urlsplit(self.server_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise TranslationError(f"无效的 Qwen server URL: {self.server_url}")
        return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)

    def _port_is_open(self) -> bool:
        host, port = self._address()
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    def _start_process(self) -> subprocess.Popen[bytes]:
        host, port = self._address()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise TranslationError(
                "只能自动启动本机 Qwen 服务；远程 server-url 当前不可连接"
            )
        command = [
            self.python_executable,
            "-m",
            "mlx_vlm.server",
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise TranslationError(f"无法启动 mlx_vlm.server: {exc}") from exc

    def ensure_ready(self) -> None:
        logger.info("检查 Qwen 服务：{}", self.server_url)
        state = self._probe()
        if state == "mlx":
            logger.info("复用已就绪的 Qwen 服务")
            return
        if state == "other" or self._port_is_open():
            _, port = self._address()
            raise TranslationError(f"端口 {port} 已被非 mlx_vlm 服务占用")

        logger.info("Qwen 服务未运行，正在自动启动：{}", self.model_path)
        self._process = self._start_process()
        self._owned = True
        deadline = self._clock() + self.start_timeout
        while self._clock() < deadline:
            if self._process.poll() is not None:
                code = self._process.returncode
                self.close()
                raise TranslationError(f"mlx_vlm.server 在就绪前退出，退出码 {code}")
            state = self._probe()
            if state == "mlx":
                logger.success("Qwen 服务已就绪")
                return
            if state == "other":
                self.close()
                raise TranslationError("Qwen 启动期间端口被其他服务占用")
            self._sleep(1.0)

        self.close()
        raise TranslationError(
            f"mlx_vlm.server 在 {self.start_timeout:g} 秒内未就绪"
        )

    def close(self) -> None:
        process = self._process
        owned = self._owned
        self._process = None
        self._owned = False
        if not owned or process is None or process.poll() is not None:
            return
        logger.info("正在关闭本次启动的 Qwen 服务")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def __enter__(self) -> QwenServiceManager:
        self.ensure_ready()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def encode_image_data_url(image: Image.Image) -> str:
    image = image.convert("RGB")
    width, height = image.size
    target_width = max(56, math.ceil(width / 28) * 28)
    target_height = max(56, math.ceil(height / 28) * 28)
    if image.size != (target_width, target_height):
        canvas = Image.new("RGB", (target_width, target_height), "white")
        canvas.paste(image, ((target_width - width) // 2, (target_height - height) // 2))
        image = canvas
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def label_target_roi(image: Image.Image, region_id: str) -> Image.Image:
    image = image.convert("RGB")
    font_size = max(18, min(36, round(image.width * 0.08)))
    font = ImageFont.load_default(size=font_size)
    label = f"TARGET ID: {region_id}"
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), label, font=font)
    label_width = right - left
    banner_height = bottom - top + 16
    canvas = Image.new(
        "RGB", (max(image.width, label_width + 16), image.height + banner_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8 - left, 8 - top), label, fill="black", font=font)
    canvas.paste(image, ((canvas.width - image.width) // 2, banner_height))
    return canvas


def validate_translation_response(
    content: str, expected_ids: Sequence[str]
) -> tuple[dict[str, Translation], set[str]]:
    expected = set(expected_ids)
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}, set(expected)
    if not isinstance(payload, dict) or set(payload) != {"items"}:
        return {}, set(expected)
    records = payload["items"]
    if not isinstance(records, list):
        return {}, set(expected)

    grouped: dict[str, list[Any]] = {}
    has_unknown_id = False
    for record in records:
        if not isinstance(record, dict):
            has_unknown_id = True
            continue
        item_id = record.get("id")
        if not isinstance(item_id, str) or item_id not in expected:
            has_unknown_id = True
            continue
        grouped.setdefault(item_id, []).append(record)
    if has_unknown_id:
        return {}, set(expected)

    valid: dict[str, Translation] = {}
    invalid: set[str] = set()
    required_keys = {"id", "source_text", "action", "translation"}
    for item_id in expected_ids:
        matches = grouped.get(item_id, [])
        if len(matches) != 1:
            invalid.add(item_id)
            continue
        record = matches[0]
        if set(record) != required_keys:
            invalid.add(item_id)
            continue
        source_text = record["source_text"]
        action = record["action"]
        translation = record["translation"]
        values_are_strings = isinstance(source_text, str) and isinstance(translation, str)
        valid_action = action in {"translate", "skip"}
        valid_payload = (
            values_are_strings
            and valid_action
            and (
                (action == "translate" and bool(source_text.strip()) and bool(translation.strip()))
                or (action == "skip" and not translation)
            )
        )
        if not valid_payload:
            invalid.add(item_id)
            continue
        valid[item_id] = Translation(item_id, source_text, action, translation)
    return valid, invalid


class QwenTranslator:
    def __init__(
        self,
        server_url: str,
        model_path: Path,
        *,
        batch_size: int = 1,
        request_timeout: float = 600.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.model_path = Path(model_path)
        self.batch_size = min(batch_size, 16)
        self.request_timeout = request_timeout

    def _post(self, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            _api_url(self.server_url, "chat/completions"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer not-needed",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "comictranslate/0.1",
            },
            method="POST",
        )
        try:
            with _open(request, timeout=self.request_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("响应 content 为空")
            return content.strip()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TranslationError(f"Qwen 请求失败: {exc}") from exc
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise TranslationError(f"Qwen HTTP 响应格式无效: {exc}") from exc

    def _request_regions(
        self,
        page_data_url: str,
        regions: Sequence[Region],
        roi_data_urls: dict[str, str],
    ) -> str:
        ids = [region.id for region in regions]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{TRANSLATION_PROMPT}\n"
                    f"本次 TARGET ID：{json.dumps(ids, ensure_ascii=False)}"
                ),
            },
        ]
        for region in regions:
            content.extend(
                [
                    {"type": "text", "text": f"目标 ROI：{region.id}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": roi_data_urls[region.id]},
                    },
                ]
            )
        content.extend(
            [
                {"type": "text", "text": "整页上下文（仅供理解语境，不得从中识别 source_text）"},
                {"type": "image_url", "image_url": {"url": page_data_url}},
            ]
        )
        payload = {
            "model": str(self.model_path),
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 4096,
            "temperature": 0.1,
            "enable_thinking": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "comic_translation",
                    "strict": True,
                    "schema": TRANSLATION_SCHEMA,
                },
            },
        }
        return self._post(payload)

    def translate(self, page_image: Image.Image, regions: Sequence[Region]) -> list[Translation]:
        if not regions:
            return []
        page_data_url = encode_image_data_url(page_image)
        roi_data_urls = {
            region.id: encode_image_data_url(
                label_target_roi(
                    page_image.crop(region.crop_bbox.as_int()), region.id
                )
            )
            for region in regions
        }
        translated: dict[str, Translation] = {}
        total_batches = math.ceil(len(regions) / self.batch_size)
        for offset in range(0, len(regions), self.batch_size):
            batch = regions[offset : offset + self.batch_size]
            batch_number = offset // self.batch_size + 1
            logger.info(
                "Qwen 翻译批次 {}/{}：{} 个区域",
                batch_number,
                total_batches,
                len(batch),
            )
            content = self._request_regions(page_data_url, batch, roi_data_urls)
            valid, invalid = validate_translation_response(
                content, [region.id for region in batch]
            )
            translated.update(valid)
            by_id = {region.id: region for region in batch}
            for item_id in sorted(invalid):
                for attempt in range(1, 3):
                    logger.warning("Qwen 响应无效，重试区域 {}（{}/2）", item_id, attempt)
                    retry_content = self._request_regions(
                        page_data_url, [by_id[item_id]], roi_data_urls
                    )
                    retry_valid, retry_invalid = validate_translation_response(
                        retry_content, [item_id]
                    )
                    if not retry_invalid:
                        translated[item_id] = retry_valid[item_id]
                        break
                else:
                    raise TranslationError(
                        f"区域 {item_id} 的 Qwen 响应连续 3 次不符合协议"
                    )
        return [translated[region.id] for region in regions]
