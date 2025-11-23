#!/usr/bin/env python

"""背景处理器

支持纯色、渐变和自定义图像背景。
"""
import cv2
import numpy as np


class BackgroundProcessor:
    """背景处理器"""

    async def add_background(
        self,
        image: np.ndarray,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """添加纯色背景

        Args:
            image: BGRA 图像
            color: 背景颜色 (BGR)

        Returns:
            BGR 图像
        """
        if image.shape[2] != 4:
            raise ValueError("Input image must have 4 channels (BGRA)")

        h, w = image.shape[:2]

        # 提取 BGR 和 Alpha
        bgr = image[:, :, :3]
        alpha = image[:, :, 3:] / 255.0

        # 创建背景
        background = np.full((h, w, 3), color, dtype=np.uint8)

        # 合成
        result = (bgr * alpha + background * (1 - alpha)).astype(np.uint8)

        return result

    async def add_gradient_background(
        self,
        image: np.ndarray,
        start_color: tuple[int, int, int],
        end_color: tuple[int, int, int],
        direction: str = "vertical",
    ) -> np.ndarray:
        """添加渐变背景

        Args:
            image: BGRA 图像
            start_color: 起始颜色 (BGR)
            end_color: 结束颜色 (BGR)
            direction: 方向 ("vertical" 或 "horizontal")

        Returns:
            BGR 图像
        """
        if image.shape[2] != 4:
            raise ValueError("Input image must have 4 channels (BGRA)")

        h, w = image.shape[:2]

        # 创建渐变背景
        background = self._create_gradient(w, h, start_color, end_color, direction)

        # 提取 BGR 和 Alpha
        bgr = image[:, :, :3]
        alpha = image[:, :, 3:] / 255.0

        # 合成
        result = (bgr * alpha + background * (1 - alpha)).astype(np.uint8)

        return result

    def _create_gradient(
        self,
        width: int,
        height: int,
        start_color: tuple[int, int, int],
        end_color: tuple[int, int, int],
        direction: str,
    ) -> np.ndarray:
        """创建渐变图像"""
        if direction == "vertical":
            # 从上到下渐变
            gradient = np.linspace(0, 1, height).reshape(height, 1, 1)
            gradient = np.tile(gradient, (1, width, 3))
        else:
            # 从左到右渐变
            gradient = np.linspace(0, 1, width).reshape(1, width, 1)
            gradient = np.tile(gradient, (height, 1, 3))

        start = np.array(start_color, dtype=np.float32)
        end = np.array(end_color, dtype=np.float32)

        background = start + gradient * (end - start)
        return background.astype(np.uint8)

    async def add_image_background(
        self,
        image: np.ndarray,
        background_image: np.ndarray,
    ) -> np.ndarray:
        """添加自定义图像背景

        Args:
            image: BGRA 图像
            background_image: 背景图像 (BGR)

        Returns:
            BGR 图像
        """
        if image.shape[2] != 4:
            raise ValueError("Input image must have 4 channels (BGRA)")

        h, w = image.shape[:2]

        # 调整背景图像大小
        if background_image.shape[:2] != (h, w):
            background_image = cv2.resize(background_image, (w, h))

        # 提取 BGR 和 Alpha
        bgr = image[:, :, :3]
        alpha = image[:, :, 3:] / 255.0

        # 合成
        result = (bgr * alpha + background_image * (1 - alpha)).astype(np.uint8)

        return result
