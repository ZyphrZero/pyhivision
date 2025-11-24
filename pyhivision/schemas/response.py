#!/usr/bin/env python

"""响应数据模型"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class FaceInfo(BaseModel):
    """人脸信息"""

    model_config = {"arbitrary_types_allowed": True}

    x: int = Field(..., description="人脸框左上角 X 坐标")
    y: int = Field(..., description="人脸框左上角 Y 坐标")
    width: int = Field(..., description="人脸框宽度")
    height: int = Field(..., description="人脸框高度")
    roll_angle: float = Field(default=0.0, description="人脸偏转角度（度）")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="检测置信度")
    landmarks: np.ndarray | None = Field(
        default=None, description="人脸关键点（68点模型，shape=(68, 2)）"
    )

    @property
    def center(self) -> tuple[int, int]:
        """人脸中心点"""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """边界框 (x1, y1, x2, y2)"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class PhotoResult(BaseModel):
    """ID Photo 处理结果"""

    model_config = {"arbitrary_types_allowed": True}

    # 输出图像
    standard: np.ndarray = Field(..., description="标准照 (BGR)")
    hd: np.ndarray | None = Field(default=None, description="高清照 (BGR)")
    matting: np.ndarray = Field(..., description="抠图结果 (BGRA)")

    # 人脸信息
    face_info: FaceInfo | None = Field(default=None, description="人脸信息")

    # 处理元数据
    processing_time_ms: float = Field(..., description="总处理时间 (毫秒)")
    stage_times: dict[str, float] = Field(
        default_factory=dict, description="各阶段耗时 (毫秒)"
    )

    # 模型信息
    model_info: dict[str, str] = Field(
        default_factory=dict, description="使用的模型信息"
    )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（不包含图像数据）"""
        return {
            "face_info": self.face_info.model_dump() if self.face_info else None,
            "processing_time_ms": self.processing_time_ms,
            "stage_times": self.stage_times,
            "model_info": self.model_info,
            "standard_shape": self.standard.shape,
            "hd_shape": self.hd.shape if self.hd is not None else None,
            "matting_shape": self.matting.shape,
        }


class ProcessingStats(BaseModel):
    """处理统计信息"""

    total_requests: int = Field(default=0, description="总请求数")
    successful_requests: int = Field(default=0, description="成功请求数")
    failed_requests: int = Field(default=0, description="失败请求数")
    avg_processing_time_ms: float = Field(default=0.0, description="平均处理时间")
    cache_hit_rate: float = Field(default=0.0, description="缓存命中率")
