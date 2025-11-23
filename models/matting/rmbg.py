#!/usr/bin/env python

"""RMBG 抠图模型实现

RMBG 使用 [-1, 1] 归一化和 1024x1024 输入尺寸。
"""
from typing import Any

import cv2
import numpy as np

from models.base import BaseMattingModel


class RMBGModel(BaseMattingModel):
    """RMBG 人像抠图模型"""

    async def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """预处理（[-1, 1] 归一化）

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

        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize 到参考尺寸（RMBG 使用 1024）
        ref_size = self.config.ref_size
        resized = cv2.resize(rgb, (ref_size, ref_size), interpolation=cv2.INTER_AREA)

        # 归一化到 [-1, 1]
        normalized = resized.astype(np.float32)
        normalized = normalized / 127.5 - 1.0

        # HWC → CHW → NCHW
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        metadata = {
            "orig_size": (orig_w, orig_h),
            "orig_image": image,
        }

        return batched, metadata

    async def postprocess(
        self, output: np.ndarray, metadata: dict[str, Any]
    ) -> np.ndarray:
        """后处理：生成 BGRA 图像

        Args:
            output: 模型输出
            metadata: 预处理元数据

        Returns:
            BGRA 图像
        """
        orig_w, orig_h = metadata["orig_size"]
        orig_image = metadata["orig_image"]

        # 提取输出
        if len(output.shape) == 4:
            matte = output[0, 0, :, :]
        else:
            matte = output[0, :, :]

        # 确保在 [0, 1] 范围内
        matte = np.clip(matte, 0, 1)

        # Resize 回原始尺寸
        matte = cv2.resize(matte, (orig_w, orig_h), interpolation=cv2.INTER_AREA)

        # 转换为 0-255
        alpha = (matte * 255).astype(np.uint8)

        # 合并为 BGRA
        bgra = np.dstack([orig_image, alpha])

        return bgra
