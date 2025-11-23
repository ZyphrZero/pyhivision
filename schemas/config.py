#!/usr/bin/env python

"""模型配置数据结构"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """AI 模型配置"""

    name: str = Field(..., description="模型名称")
    checkpoint_path: Path = Field(..., description="权重文件路径")
    ref_size: int = Field(default=512, ge=256, le=2048, description="参考尺寸")
    use_gpu: bool = Field(default=False, description="是否使用 GPU")

    model_config = {"arbitrary_types_allowed": True}


class MattingModelConfig(ModelConfig):
    """抠图模型配置"""

    model_type: Literal["modnet", "birefnet", "rmbg"] = Field(
        default="modnet", description="模型类型"
    )
    # ModNet 归一化: [-1, 1]
    # BiRefNet/RMBG 归一化: ImageNet 标准
    normalize_mode: Literal["modnet", "imagenet"] = Field(
        default="modnet", description="归一化模式"
    )


class DetectionModelConfig(ModelConfig):
    """检测模型配置"""

    model_type: Literal["mtcnn", "retinaface"] = Field(
        default="mtcnn", description="模型类型"
    )
    conf_threshold: float = Field(default=0.8, ge=0.1, le=1.0, description="置信度阈值")
