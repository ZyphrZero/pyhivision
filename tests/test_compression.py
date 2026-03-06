import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyhivision.utils.compression import compress_to_kb, compress_to_kb_base64, save_with_dpi

TARGET_KB = 30
TARGET_BYTES = TARGET_KB * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _random_bgr_image(height: int = 256, width: int = 256) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _decode_base64_data_url(data_url: str) -> bytes:
    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def test_compress_to_kb_exact_size_and_output_file(tmp_path: Path):
    image = _random_bgr_image()
    output_file = tmp_path / "compressed.jpg"

    data = compress_to_kb(image=image, target_kb=TARGET_KB, output_path=str(output_file))

    assert len(data) == TARGET_BYTES
    assert data.startswith(b"\xff\xd8")
    assert output_file.exists()
    assert output_file.read_bytes() == data


def test_compress_to_kb_rejects_unsupported_shape():
    bad_shape = np.zeros((10, 10, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported image shape"):
        compress_to_kb(bad_shape, target_kb=10)


def test_compress_to_kb_base64_modes_follow_size_contract():
    image = _random_bgr_image()

    exact = compress_to_kb_base64(image, target_kb=TARGET_KB, mode="exact")
    max_mode = compress_to_kb_base64(image, target_kb=TARGET_KB, mode="max")
    min_mode = compress_to_kb_base64(image, target_kb=TARGET_KB, mode="min")

    assert exact.startswith("data:image/jpeg;base64,")
    assert max_mode.startswith("data:image/jpeg;base64,")
    assert min_mode.startswith("data:image/jpeg;base64,")

    exact_bytes = _decode_base64_data_url(exact)
    max_bytes = _decode_base64_data_url(max_mode)
    min_bytes = _decode_base64_data_url(min_mode)

    assert len(exact_bytes) == TARGET_BYTES
    assert len(max_bytes) <= TARGET_BYTES
    assert len(min_bytes) >= TARGET_BYTES


def test_compress_to_kb_base64_rejects_invalid_input():
    with pytest.raises(ValueError, match="image must be a NumPy array or PIL Image"):
        compress_to_kb_base64("not_an_image", target_kb=10)


def test_save_with_dpi_png_bytes_and_metadata(tmp_path: Path):
    image = _random_bgr_image(64, 64)
    output_file = tmp_path / "dpi_image.png"

    data = save_with_dpi(image=image, dpi=300, output_path=str(output_file))

    assert data.startswith(PNG_SIGNATURE)
    assert output_file.exists()
    assert output_file.read_bytes() == data

    loaded = Image.open(io.BytesIO(data))
    dpi = loaded.info.get("dpi")
    assert dpi is not None
    assert dpi[0] == pytest.approx(300, abs=1)
    assert dpi[1] == pytest.approx(300, abs=1)


def test_save_with_dpi_supports_bgra_input():
    bgra = np.zeros((32, 32, 4), dtype=np.uint8)
    bgra[:, :, 3] = 255

    data = save_with_dpi(bgra, dpi=200)
    loaded = Image.open(io.BytesIO(data))

    assert loaded.mode == "RGBA"
