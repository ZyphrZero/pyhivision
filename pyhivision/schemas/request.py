#!/usr/bin/env python

"""请求数据模型"""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from pyhivision.exceptions.errors import ImageTooSmallError
from pyhivision.schemas.alignment import AlignmentParams

# 钩子函数类型定义
HookFunction = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]


class BeautyParams(BaseModel):
    """美颜参数"""

    # Basic adjustments
    brightness: int = Field(default=0, ge=-100, le=100, description="亮度调整")
    contrast: int = Field(default=0, ge=-100, le=100, description="对比度调整")
    sharpen: int = Field(default=0, ge=0, le=100, description="锐化程度")
    saturation: int = Field(default=0, ge=-100, le=100, description="饱和度调整")

    # Advanced beauty features
    whitening: int = Field(default=0, ge=0, le=30, description="美白强度")
    skin_smoothing: int = Field(default=0, ge=0, le=10, description="磨皮强度")
    grind_degree: int = Field(default=3, ge=1, le=10, description="磨皮程度")
    detail_degree: int = Field(default=1, ge=1, le=10, description="细节保留程度")

    @property
    def is_enabled(self) -> bool:
        """是否需要美颜处理"""
        return any(
            [
                self.brightness,
                self.contrast,
                self.sharpen,
                self.saturation,
                self.whitening,
                self.skin_smoothing,
            ]
        )


class LayoutParams(BaseModel):
    """布局参数"""

    head_measure_ratio: float = Field(
        default=0.2, ge=0.1, le=0.5, description="头部比例"
    )
    head_height_ratio: float = Field(
        default=0.45, ge=0.3, le=0.7, description="头顶到图片顶部的比例"
    )
    top_distance_max: float = Field(
        default=0.12, ge=0.05, le=0.3, description="头顶最大距离比例"
    )
    top_distance_min: float = Field(
        default=0.1, ge=0.03, le=0.2, description="头顶最小距离比例"
    )

    # 高级参数
    horizontal_flip: bool = Field(default=False, description="是否水平翻转照片")
    clothing_type: str | None = Field(default=None, description="换装类型（future use）")
    clothing_color: tuple[int, int, int] | None = Field(
        default=None, description="换装颜色 (BGR) (future use)"
    )


class PhotoRequest(BaseModel):
    """ID Photo 处理请求"""

    model_config = {"arbitrary_types_allowed": True}

    # 输入图像
    image: np.ndarray = Field(..., description="输入图像 (BGR 格式)")

    # 目标尺寸 (高度, 宽度)
    size: tuple[int, int] = Field(default=(413, 295), description="目标尺寸 (高度, 宽度)")

    # 背景颜色 (BGR 格式)
    background_color: tuple[int, int, int] = Field(
        default=(255, 255, 255), description="背景颜色 (BGR)"
    )

    # 模型选择
    matting_model: Literal[
        "modnet_photographic",
        "hivision_modnet",
        "birefnet_lite",
        "rmbg_1.4",
        "rmbg_2.0",
    ] = Field(default="modnet_photographic", description="抠图模型")

    detection_model: Literal["mtcnn", "retinaface"] = Field(
        default="mtcnn", description="人脸检测模型"
    )

    # 可选参数
    beauty_params: BeautyParams = Field(
        default_factory=BeautyParams, description="美颜参数"
    )
    layout_params: LayoutParams = Field(
        default_factory=LayoutParams, description="布局参数"
    )
    alignment_params: AlignmentParams = Field(
        default_factory=AlignmentParams, description="人脸矫正参数"
    )

    # 处理选项
    change_bg_only: bool = Field(default=False, description="仅换背景（跳过人脸检测）")
    render_hd: bool = Field(default=True, description="是否生成高清照")
    render_matting: bool = Field(default=False, description="是否返回抠图结果")
    enable_matting_fix: bool = Field(
        default=False, description="是否启用抠图修补（仅对 hivision_modnet 有效）"
    )
    add_background: bool = Field(
        default=True, description="是否添加背景色（False 则返回透明背景的 BGRA 图像）"
    )

    # 检测选项
    detection_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="人脸检测置信度阈值（低于此值的检测结果将被过滤）",
    )
    detection_nms_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="NMS IoU 阈值（用于过滤重叠的检测框）",
    )
    multiple_faces_strategy: Literal["error", "best", "largest"] = Field(
        default="best",
        description=(
            "多人脸处理策略："
            "error - 检测到多人脸时报错（严格模式）；"
            "best - 选择置信度最高的人脸；"
            "largest - 选择面积最大的人脸"
        ),
    )

    # 处理流程钩子
    hooks: dict[str, HookFunction] = Field(
        default_factory=dict,
        description=(
            "处理流程钩子函数，支持：after_matting, after_detection, "
            "after_beauty, after_adjustment, after_background"
        ),
    )

    @field_validator("image", mode="before")
    @classmethod
    def validate_image(cls, v: np.ndarray) -> np.ndarray:
        """验证图像"""
        if not isinstance(v, np.ndarray):
            raise ValueError("Image must be a numpy array")

        if len(v.shape) not in (2, 3):
            raise ValueError(f"Invalid image dimensions: {len(v.shape)}")

        h, w = v.shape[:2]
        min_size = 100

        if h < min_size or w < min_size:
            raise ImageTooSmallError(w, h, min_size)

        return v

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        """验证目标尺寸"""
        height, width = v
        if height < 100 or width < 100:
            raise ValueError(f"Size too small: {v}")
        if height > 2000 or width > 2000:
            raise ValueError(f"Size too large: {v}")
        return v

    @field_validator("background_color")
    @classmethod
    def validate_background_color(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        """验证背景颜色"""
        for i, c in enumerate(v):
            if not 0 <= c <= 255:
                raise ValueError(f"Invalid color value at index {i}: {c}")
        return v
