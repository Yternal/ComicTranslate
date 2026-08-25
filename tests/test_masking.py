import numpy as np
import pytest

from comictranslate.errors import MaskError
from comictranslate.masking import stitch_masks
from comictranslate.models import BBox, Region


def test_stitch_mask_binarizes_and_dilates_once() -> None:
    region = Region("a", "free", BBox(2, 2, 3, 3), BBox(1, 1, 4, 4))
    roi = np.zeros((3, 3), np.uint8)
    roi[1, 1] = 200
    result = stitch_masks((5, 5), [region], {"a": roi})
    assert np.all(result[1:4, 1:4] == 255)
    assert result[0, 0] == 0


def test_stitch_mask_rejects_wrong_roi_size() -> None:
    region = Region("a", "free", BBox(2, 2, 3, 3), BBox(1, 1, 4, 4))
    with pytest.raises(MaskError, match="尺寸"):
        stitch_masks((5, 5), [region], {"a": np.zeros((2, 2), np.uint8)})


def test_empty_region_list_produces_empty_global_mask() -> None:
    result = stitch_masks((7, 9), [], {})
    assert result.shape == (9, 7)
    assert not result.any()
