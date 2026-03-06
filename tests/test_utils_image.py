import numpy as np
import pytest

from pyhivision.utils.image import get_content_box


def test_get_content_box_rejects_non_bgra_input():
    gray = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(TypeError, match="四通道"):
        get_content_box(gray)

