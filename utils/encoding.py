#!/usr/bin/env python

"""图像编码/解码工具

提供 NumPy 数组与 Base64、字节流之间的转换功能。

"""

import base64
import io

import cv2
import numpy as np
from PIL import Image


def numpy_to_base64(image: np.ndarray, format: str = "png") -> str:
    """NumPy 数组转 Base64 字符串

    Args:
        image: 输入图像数组（BGR 或 BGRA 格式）
        format: 图像格式，可选 'png' 或 'jpeg'

    Returns:
        Base64 字符串，包含 data URL 前缀（如 "data:image/png;base64,..."）

    Raises:
        ValueError: 图像格式不支持

    Examples:
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> b64 = numpy_to_base64(image)
        >>> assert b64.startswith("data:image/png;base64,")
    """
    if format not in ("png", "jpeg"):
        raise ValueError(f"Unsupported format: {format}, must be 'png' or 'jpeg'")

    # 使用 OpenCV 编码图像
    ext = f".{format}"
    success, buffer = cv2.imencode(ext, image)

    if not success:
        raise ValueError("Failed to encode image")

    # 转换为 Base64
    base64_str = base64.b64encode(buffer).decode("utf-8")

    return f"data:image/{format};base64,{base64_str}"


def base64_to_numpy(base64_string: str) -> np.ndarray:
    """Base64 字符串转 NumPy 数组

    Args:
        base64_string: Base64 编码的图像字符串（支持 data URL 格式）

    Returns:
        NumPy 图像数组（BGR 或 BGRA 格式）

    Raises:
        ValueError: Base64 解码失败或图像解码失败

    Examples:
        >>> b64 = "data:image/png;base64,iVBORw0KGgo..."
        >>> image = base64_to_numpy(b64)
        >>> assert isinstance(image, np.ndarray)
    """
    # 去除 data URL 前缀（如果存在）
    if base64_string.startswith("data:image"):
        # 格式: "data:image/png;base64,<base64-data>"
        base64_string = base64_string.split(",", 1)[1]

    # Base64 解码为字节
    try:
        img_bytes = base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 string: {e}") from e

    # 字节转 NumPy 数组
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)

    # 使用 OpenCV 解码图像
    image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("Failed to decode image from bytes")

    return image


def bytes_to_base64(data: bytes) -> str:
    """字节流转 Base64 字符串

    Args:
        data: 图像字节数据

    Returns:
        Base64 字符串，包含 data URL 前缀

    Examples:
        >>> with open("image.png", "rb") as f:
        ...     data = f.read()
        >>> b64 = bytes_to_base64(data)
        >>> assert b64.startswith("data:image/png;base64,")
    """
    base64_str = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def numpy_to_bytes(image: np.ndarray, format: str = "png") -> bytes:
    """NumPy 数组转字节流

    Args:
        image: 输入图像数组
        format: 输出格式，可选 'png' 或 'jpeg'

    Returns:
        图像字节数据

    Raises:
        ValueError: 图像格式不支持或编码失败

    Examples:
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> data = numpy_to_bytes(image)
        >>> assert isinstance(data, bytes)
    """
    if format not in ("png", "jpeg"):
        raise ValueError(f"Unsupported format: {format}")

    # 转换为 PIL Image
    # 需要处理通道顺序：OpenCV 使用 BGR，PIL 使用 RGB
    if len(image.shape) == 2:
        # 灰度图
        pil_image = Image.fromarray(image, mode="L")
    elif image.shape[2] == 3:
        # BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image, mode="RGB")
    elif image.shape[2] == 4:
        # BGRA -> RGBA
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        pil_image = Image.fromarray(rgba_image, mode="RGBA")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # 保存到字节流
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format=format.upper())

    return byte_arr.getvalue()


def save_numpy_image(image: np.ndarray, file_path: str) -> None:
    """保存 NumPy 图像到文件

    自动处理 BGR/BGRA 到 RGB/RGBA 的转换（OpenCV → PIL）。

    Args:
        image: 图像数组（BGR 或 BGRA 格式）
        file_path: 保存路径

    Raises:
        ValueError: 图像格式不支持
        IOError: 文件保存失败

    Examples:
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> save_numpy_image(image, "output.png")
    """
    if len(image.shape) == 2:
        # 灰度图
        pil_image = Image.fromarray(image, mode="L")
    elif image.shape[2] == 3:
        # BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image, mode="RGB")
    elif image.shape[2] == 4:
        # BGRA -> RGBA
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        pil_image = Image.fromarray(rgba_image, mode="RGBA")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    try:
        pil_image.save(file_path)
    except Exception as e:
        raise IOError(f"Failed to save image to {file_path}: {e}") from e
