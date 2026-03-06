import numpy as np

from pyhivision.schemas.response import PhotoResult


def test_photo_result_to_dict_handles_none_optional_images():
    result = PhotoResult(standard=np.zeros((8, 8, 3), dtype=np.uint8))

    payload = result.to_dict()

    assert payload["standard_shape"] == (8, 8, 3)
    assert payload["hd_shape"] is None
    assert payload["matting_shape"] is None

