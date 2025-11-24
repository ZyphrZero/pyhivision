#!/usr/bin/env python

"""图像压缩工具

提供图像压缩到指定 KB 大小和 DPI 设置功能。

"""

import io
from typing import Literal

import numpy as np
from PIL import Image


def compress_to_kb(
    image: np.ndarray,
    target_kb: int,
    dpi: int = 300,
    output_path: str | None = None,
) -> bytes:
    """压缩图像到指定 KB 大小

    算法：循环降低 JPEG quality 直到满足大小要求，不足则填充字节。

    Args:
        image: 输入图像（NumPy 数组，BGR 格式）
        target_kb: 目标文件大小（KB）
        dpi: DPI 设置（默认 300）
        output_path: 可选保存路径

    Returns:
        压缩后的图像字节数据（JPEG 格式）

    Note:
        - 使用 JPEG 格式压缩
        - 质量从 95 逐步降低到 1
        - 小于目标大小时填充 0x00 字节

    Examples:
        >>> image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        >>> data = compress_to_kb(image, target_kb=50)
        >>> assert 49 * 1024 <= len(data) <= 51 * 1024
    """
    # 转换为 PIL Image（OpenCV BGR -> PIL RGB）
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            # 灰度图
            pil_image = Image.fromarray(image, mode="L")
        elif image.shape[2] == 3:
            # BGR -> RGB
            import cv2

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image, mode="RGB")
        elif image.shape[2] == 4:
            # BGRA -> RGB (丢弃 alpha 通道，JPEG 不支持透明度)
            import cv2

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            pil_image = Image.fromarray(rgb_image, mode="RGB")
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError("image must be a NumPy array or PIL Image")

    # 确保 RGB 模式
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # 初始质量值
    quality = 95

    while True:
        # 创建字节流
        img_byte_arr = io.BytesIO()

        # 保存图像（带 DPI 元数据）
        pil_image.save(img_byte_arr, format="JPEG", quality=quality, dpi=(dpi, dpi))

        # 获取当前大小（KB）
        img_size_kb = len(img_byte_arr.getvalue()) / 1024

        # 检查是否满足要求
        if img_size_kb <= target_kb or quality == 1:
            # 如果小于目标大小，填充字节
            if img_size_kb < target_kb:
                padding_size = int((target_kb * 1024) - len(img_byte_arr.getvalue()))
                padding = b"\x00" * padding_size
                img_byte_arr.write(padding)

            # 保存到文件（如果指定了路径）
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(img_byte_arr.getvalue())

            return img_byte_arr.getvalue()

        # 降低质量
        quality -= 5

        # 确保质量不低于 1
        if quality < 1:
            quality = 1


def compress_to_kb_base64(
    image: np.ndarray,
    target_kb: int,
    mode: Literal["exact", "max", "min"] = "exact",
) -> str:
    """压缩图像到指定 KB 并返回 Base64

    Args:
        image: 输入图像（NumPy 数组）
        target_kb: 目标大小（KB）
        mode: 压缩模式
            - exact: 精确到目标大小（填充字节）
            - max: 不大于目标大小
            - min: 不小于目标大小

    Returns:
        Base64 字符串，包含 data URL 前缀

    Examples:
        >>> image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        >>> b64 = compress_to_kb_base64(image, target_kb=30, mode="max")
        >>> assert b64.startswith("data:image/jpeg;base64,")
    """
    import base64

    import cv2

    # 转换为 PIL Image
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image, mode="L")
        elif image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image, mode="RGB")
        elif image.shape[2] == 4:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            pil_image = Image.fromarray(rgb_image, mode="RGB")
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError("image must be a NumPy array or PIL Image")

    # 确保 RGB 模式
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # 初始质量值
    quality = 95

    while True:
        # 创建字节流
        img_byte_arr = io.BytesIO()

        # 保存图像
        pil_image.save(img_byte_arr, format="JPEG", quality=quality)

        # 获取当前大小（KB）
        img_size_kb = len(img_byte_arr.getvalue()) / 1024

        # 根据模式检查
        if mode == "exact":
            # 精确大小
            if img_size_kb == target_kb:
                break
            elif img_size_kb < target_kb:
                # 填充字节
                padding_size = int((target_kb * 1024) - len(img_byte_arr.getvalue()))
                padding = b"\x00" * padding_size
                img_byte_arr.write(padding)
                break

        elif mode == "max":
            # 不大于目标大小
            if img_size_kb <= target_kb or quality == 1:
                break

        elif mode == "min" and img_size_kb >= target_kb:
            # 不小于目标大小
            break

        # 降低质量
        quality -= 5

        # 确保质量不低于 1
        if quality < 1:
            quality = 1

    # 转换为 Base64
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


def save_with_dpi(
    image: np.ndarray,
    dpi: int = 300,
    output_path: str | None = None,
) -> bytes:
    """设置 DPI 并保存图像

    Args:
        image: 输入图像（NumPy 数组，BGR 格式）
        dpi: DPI 值（默认 300）
        output_path: 可选保存路径

    Returns:
        PNG 格式字节数据（包含 DPI 元数据）

    Examples:
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> data = save_with_dpi(image, dpi=300)
        >>> assert isinstance(data, bytes)
    """
    import cv2

    # 转换为 PIL Image
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

    # 创建字节流
    byte_stream = io.BytesIO()

    # 保存为 PNG（带 DPI 元数据）
    pil_image.save(byte_stream, format="PNG", dpi=(dpi, dpi))

    # 获取字节数据
    image_bytes = byte_stream.getvalue()

    # 保存到文件（如果指定了路径）
    if output_path:
        with open(output_path, "wb") as f:
            f.write(image_bytes)

    return image_bytes
