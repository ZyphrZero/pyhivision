import numpy as np
import pytest
from pydantic import ValidationError

from pyhivision.schemas.request import PhotoRequest
from pyhivision.utils.image import parse_color_to_bgr


def test_photo_request_rejects_removed_color_format_field():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PhotoRequest(
            image=image,
            size=(413, 295),
            background_color=(255, 0, 0),
            color_format="BGR",
        )


def test_parse_color_to_bgr_accepts_only_rgb_tuple_and_hex():
    assert parse_color_to_bgr((255, 0, 0)) == (0, 0, 255)
    assert parse_color_to_bgr("#00FF00") == (0, 255, 0)
    assert parse_color_to_bgr("0000FF") == (255, 0, 0)

