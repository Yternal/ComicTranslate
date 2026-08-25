import json

import pytest
from PIL import Image

from comictranslate.errors import TranslationError
from comictranslate.models import BBox, Region
from comictranslate.qwen import (
    QwenTranslator,
    label_target_roi,
    validate_translation_response,
)


def _response(*items: dict[str, str]) -> str:
    return json.dumps({"items": list(items)}, ensure_ascii=False)


def _item(item_id: str, action: str = "translate") -> dict[str, str]:
    return {
        "id": item_id,
        "source_text": "hello" if action == "translate" else "中文",
        "action": action,
        "translation": "你好" if action == "translate" else "",
    }


def test_valid_translation_response() -> None:
    valid, invalid = validate_translation_response(
        _response(_item("a"), _item("b", "skip")), ["a", "b"]
    )
    assert not invalid
    assert valid["a"].translation == "你好"
    assert valid["b"].action == "skip"


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        _response(_item("a"), _item("a")),
        _response(_item("unknown")),
        json.dumps({"items": [{**_item("a"), "extra": "x"}]}),
        _response({**_item("a"), "translation": ""}),
        _response({**_item("a", "skip"), "translation": "不应存在"}),
    ],
)
def test_invalid_translation_response(content: str) -> None:
    _valid, invalid = validate_translation_response(content, ["a"])
    assert invalid == {"a"}


class StubTranslator(QwenTranslator):
    def __init__(self, responses: list[str]) -> None:
        super().__init__("http://127.0.0.1:8080/v1", "/model", batch_size=16)
        self.responses = iter(responses)
        self.calls: list[list[str]] = []

    def _request_regions(self, page_data_url, regions, roi_data_urls):  # type: ignore[no-untyped-def]
        self.calls.append([region.id for region in regions])
        return next(self.responses)


def _regions() -> list[Region]:
    return [
        Region("a", "free", BBox(1, 1, 10, 10), BBox(0, 0, 12, 12)),
        Region("b", "free", BBox(12, 1, 20, 10), BBox(10, 0, 22, 12)),
    ]


def test_target_roi_has_visible_id_banner() -> None:
    roi = Image.new("RGB", (80, 40), "red")

    labeled = label_target_roi(roi, "region-0001")

    assert labeled.width >= roi.width
    assert labeled.height > roi.height
    assert labeled.getpixel((labeled.width // 2, labeled.height - 1)) == (255, 0, 0)


def test_request_places_target_roi_before_page_context() -> None:
    class PayloadTranslator(QwenTranslator):
        def _post(self, payload):  # type: ignore[no-untyped-def]
            self.payload = payload
            return _response(_item("a"))

    translator = PayloadTranslator("http://127.0.0.1:8080/v1", "/model")
    translator._request_regions("page-url", [_regions()[0]], {"a": "roi-url"})
    content = translator.payload["messages"][0]["content"]
    image_urls = [
        item["image_url"]["url"] for item in content if item["type"] == "image_url"
    ]

    assert image_urls == ["roi-url", "page-url"]
    assert "禁止从整页中另选文字" in content[0]["text"]


def test_missing_id_is_retried_as_single_roi() -> None:
    translator = StubTranslator([_response(_item("a")), _response(_item("b"))])
    result = translator.translate(Image.new("RGB", (24, 16), "white"), _regions())
    assert [item.id for item in result] == ["a", "b"]
    assert translator.calls == [["a", "b"], ["b"]]


def test_invalid_single_roi_fails_after_two_retries() -> None:
    translator = StubTranslator(["bad", "bad", "bad"])
    with pytest.raises(TranslationError, match="连续 3 次"):
        translator.translate(Image.new("RGB", (24, 16), "white"), [_regions()[0]])
    assert translator.calls == [["a"], ["a"], ["a"]]


def test_batches_never_exceed_sixteen_regions() -> None:
    regions = [
        Region(
            f"r{index}",
            "free",
            BBox(index, 0, index + 1, 1),
            BBox(index, 0, index + 1, 1),
        )
        for index in range(17)
    ]
    responses = [
        _response(*[_item(region.id) for region in regions[:16]]),
        _response(_item(regions[16].id)),
    ]
    translator = StubTranslator(responses)
    translator.translate(Image.new("RGB", (20, 2), "white"), regions)
    assert [len(call) for call in translator.calls] == [16, 1]
