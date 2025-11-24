#!/usr/bin/env python

"""ModNet 抠图模型实现

避免 PIL 转换，使用 NumPy 原地操作减少内存拷贝。
"""
from typing import Any

import cv2
import numpy as np

from pyhivision.models.base import BaseMattingModel


class ModNetModel(BaseMattingModel):
    """ModNet 人像抠图模型"""

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """预处理

        Args:
            image: 输入图像 (BGR 格式)

        Returns:
            模型输入和元数据
        """
        orig_h, orig_w = image.shape[:2]

        # 确保是 BGR 3 通道
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Resize 到参考尺寸
        ref_size = self.config.ref_size
        resized = cv2.resize(image, (ref_size, ref_size), interpolation=cv2.INTER_AREA)

        # 归一化 [-1, 1]（NumPy 原地操作）
        normalized = resized.astype(np.float32)
        normalized -= 127.5
        normalized /= 127.5

        # HWC → CHW → NCHW
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        metadata = {
            "orig_size": (orig_w, orig_h),
            "orig_image": image,
        }

        return batched, metadata

    def postprocess(
        self, output: np.ndarray, metadata: dict[str, Any]
    ) -> np.ndarray:
        """后处理：生成 BGRA 图像

        Args:
            output: 模型输出 (1, 1, H, W) 或 (1, H, W)
            metadata: 预处理元数据

        Returns:
            BGRA 图像
        """
        orig_w, orig_h = metadata["orig_size"]
        orig_image = metadata["orig_image"]

        # 提取 alpha 通道
        matte = output[0, 0, :, :] if len(output.shape) == 4 else output[0, :, :]

        # Resize 回原始尺寸
        matte = cv2.resize(matte, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

        # 转换为 0-255
        alpha = (matte * 255).astype(np.uint8)

        # 合并为 BGRA
        bgra = np.dstack([orig_image, alpha])

        return bgra


class ModNetPhotographicModel(ModNetModel):
    """ModNet Photographic Portrait Matting 模型

    与基础 ModNet 使用相同的预处理和后处理逻辑。
    """

    pass


class HivisionModNetModel(ModNetModel):
    """HiVision 自定义 ModNet 模型

    与基础 ModNet 使用相同的预处理和后处理逻辑。
    """

    pass
