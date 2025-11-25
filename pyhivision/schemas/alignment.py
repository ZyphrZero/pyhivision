#!/usr/bin/env python

"""人脸矫正数据模型

定义人脸矫正相关的参数和结果模型。
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pyhivision.schemas.response import FaceInfo


class AlignmentParams(BaseModel):
    """人脸矫正参数"""

    model_config = {"arbitrary_types_allowed": True}

    enable_alignment: bool = Field(
        default=True,
        description="是否启用人脸角度矫正"
    )

    angle_threshold: float = Field(
        default=2.0,
        ge=0.0,
        le=45.0,
        description="触发矫正的最小角度阈值（度）"
    )

    rotation_center: tuple[int, int] | None = Field(
        default=None,
        description="旋转中心坐标 (x, y)，None 表示使用图像中心"
    )

    interpolation: Literal["linear", "cubic", "nearest"] = Field(
        default="linear",
        description="插值方法：linear(线性)、cubic(三次)、nearest(最近邻)"
    )

    border_mode: Literal["constant", "replicate", "reflect"] = Field(
        default="constant",
        description="边界填充模式：constant(常量)、replicate(复制)、reflect(反射)"
    )

    border_value: tuple[int, int, int, int] = Field(
        default=(0, 0, 0, 0),
        description="边界填充值 (B, G, R, A)，仅在 border_mode='constant' 时有效"
    )


class AlignmentResult(BaseModel):
    """人脸矫正结果"""

    model_config = {"arbitrary_types_allowed": True}

    rotated_image: np.ndarray = Field(
        ...,
        description="旋转后的 RGB 图像 (BGR 格式)"
    )

    rotated_alpha: np.ndarray | None = Field(
        default=None,
        description="旋转后的 Alpha 通道（单通道灰度图）"
    )

    rotation_angle: float = Field(
        ...,
        description="实际应用的旋转角度（度）"
    )

    rotation_matrix: np.ndarray = Field(
        ...,
        description="旋转变换矩阵 (2x3)"
    )

    offset_width: int = Field(
        ...,
        description="宽度偏移量（新画布宽度 - 原画布宽度）"
    )

    offset_height: int = Field(
        ...,
        description="高度偏移量（新画布高度 - 原画布高度）"
    )

    is_rotated: bool = Field(
        ...,
        description="是否实际执行了旋转操作"
    )

    @property
    def new_size(self) -> tuple[int, int]:
        """新画布尺寸 (width, height)"""
        h, w = self.rotated_image.shape[:2]
        return (w, h)

    @property
    def original_size(self) -> tuple[int, int]:
        """原始画布尺寸 (width, height)"""
        h, w = self.rotated_image.shape[:2]
        return (w - self.offset_width, h - self.offset_height)

    def transform_face_info(self, face_info: "FaceInfo") -> "FaceInfo":
        """通过旋转矩阵变换人脸坐标

        Args:
            face_info: 原始人脸信息

        Returns:
            变换后的人脸信息
        """
        from pyhivision.schemas.response import FaceInfo

        # 如果未旋转，直接返回原始信息
        if not self.is_rotated:
            return face_info

        # 人脸框四个角点
        corners = np.array([
            [face_info.x, face_info.y],
            [face_info.x + face_info.width, face_info.y],
            [face_info.x, face_info.y + face_info.height],
            [face_info.x + face_info.width, face_info.y + face_info.height],
        ], dtype=np.float32)

        # 应用旋转变换 (x, y, 1) @ M.T
        ones = np.ones((4, 1), dtype=np.float32)
        corners_homo = np.hstack([corners, ones])
        transformed = corners_homo @ self.rotation_matrix.T

        # 计算新的边界框
        x_min, y_min = transformed.min(axis=0)
        x_max, y_max = transformed.max(axis=0)

        return FaceInfo(
            x=int(x_min),
            y=int(y_min),
            width=int(x_max - x_min),
            height=int(y_max - y_min),
            roll_angle=0.0,  # 矫正后角度为 0
            confidence=face_info.confidence,
            landmarks=None,  # landmarks 需要单独变换，暂不支持
        )
