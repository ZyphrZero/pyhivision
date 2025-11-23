#!/usr/bin/env python

"""IDPhoto SDK 配置管理

使用 Pydantic Settings 实现类型安全的配置管理,
支持从环境变量和 .env 文件加载配置。

用法:
    from pyhivision import create_settings, IDPhotoSDK

    # 使用默认配置
    settings = create_settings()

    # 覆盖特定配置
    settings = create_settings(enable_gpu=True, models_dir="/custom/path")

    # 创建管线时传入配置
    pipeline = IDPhotoSDK.create(settings=settings)
"""
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ========== 默认模型文件名配置 ==========

DEFAULT_MATTING_MODEL_FILES = {
    "modnet_photographic": "modnet_photographic_portrait_matting.onnx",
    "hivision_modnet": "hivision_modnet.onnx",
    "birefnet_lite": "birefnet-v1-lite.onnx",
    "rmbg_1_4": "rmbg-1.4.onnx",
}

DEFAULT_DETECTION_MODEL_FILES = {
    "mtcnn": None,  # MTCNN 使用内置权重
    "retinaface": "retinaface-resnet50.onnx",
}


class HivisionSettings(BaseSettings):
    """IDPhoto SDK 配置类

    所有配置项均可通过环境变量覆盖（使用 HIVISION_ 前缀）。
    """

    model_config = SettingsConfigDict(
        env_prefix="HIVISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    # 性能配置
    enable_gpu: bool = Field(default=False, description="是否启用 GPU 加速")
    num_threads: int = Field(default=4, ge=1, le=32, description="ONNX Runtime 线程数")
    model_cache_size: int = Field(default=3, ge=1, le=10, description="模型缓存数量上限")
    model_ttl_minutes: int = Field(default=10, ge=1, le=60, description="模型缓存 TTL (分钟)")

    # 图像处理配置
    max_image_size: int = Field(default=2000, ge=500, le=4000, description="图像最大边长")
    matting_ref_size: int = Field(default=512, description="抠图模型参考尺寸")
    hd_photo_min_size: int = Field(default=600, ge=300, le=1200, description="高清照片最小边长")

    # 日志配置
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="日志级别"
    )
    log_format: Literal["text", "json"] = Field(default="text", description="日志格式")

    # 缓存配置
    enable_cache: bool = Field(default=True, description="是否启用结果缓存")
    cache_backend: Literal["memory", "redis"] = Field(default="memory", description="缓存后端")
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="缓存 TTL (秒)")

    # 模型路径配置（由上层应用控制，SDK 不提供默认路径）
    matting_models_dir: Path | str | None = Field(
        default=None,
        description=(
            "抠图模型权重目录（由上层应用提供，可通过环境变量 HIVISION_MATTING_MODELS_DIR 设置）\n"
            "建议路径：~/.pyhivision/models/matting 或应用程序数据目录"
        ),
    )
    detection_models_dir: Path | str | None = Field(
        default=None,
        description=(
            "检测模型权重目录（由上层应用提供，可通过环境变量 HIVISION_DETECTION_MODELS_DIR 设置）\n"
            "建议路径：~/.pyhivision/models/detection 或应用程序数据目录\n"
            "注意：MTCNN 使用内置权重，不需要此目录"
        ),
    )

    # 模型文件名映射配置
    matting_model_files: dict[str, str] = Field(
        default_factory=lambda: DEFAULT_MATTING_MODEL_FILES.copy(),
        description="抠图模型文件名映射 (model_name -> filename)",
    )
    detection_model_files: dict[str, str | None] = Field(
        default_factory=lambda: DEFAULT_DETECTION_MODEL_FILES.copy(),
        description="检测模型文件名映射 (model_name -> filename, None 表示使用内置权重)",
    )

    # 资源路径配置 (可选，由上层应用控制)
    # ⚠️ 除 LUT 外，其他资源由上层应用提供
    watermark_font_path: Path | None = Field(
        default=None,
        description="水印默认字体路径 (可选，如不提供则需在调用时传入)",
    )
    beauty_lut_path: Path | None = Field(
        default=None,
        description="自定义美颜 LUT 图像路径 (可选，默认使用 SDK 内置 LUT)",
    )
    templates_dir: Path | None = Field(
        default=None,
        description="模板资源目录路径 (可选，如不提供则禁用模板功能)",
    )

    @field_validator("matting_models_dir", "detection_models_dir", mode="before")
    @classmethod
    def validate_model_dirs(cls, v):
        """验证并转换模型目录路径

        支持字符串路径转换为 Path 对象。
        支持波浪线 (~) 扩展为用户主目录。
        允许 None 值（由上层应用在运行时提供）。
        """
        if v is None:
            return None
        if isinstance(v, str):
            # 展开用户主目录
            expanded = Path(v).expanduser()
            return expanded
        if isinstance(v, Path):
            return v.expanduser()
        # 处理其他可能的路径类型
        return Path(str(v)).expanduser()

    @field_validator("watermark_font_path", "beauty_lut_path", "templates_dir", mode="before")
    @classmethod
    def validate_resource_paths(cls, v):
        """验证并转换资源路径

        资源路径是可选的，由上层应用控制
        """
        if v is None:
            return None

        if isinstance(v, str):
            return Path(v)

        if not isinstance(v, Path):
            return Path(v)

        return v


# ========== 工厂函数 ==========


def create_settings(**overrides) -> HivisionSettings:
    """创建配置实例

    Args:
        **overrides: 配置覆盖参数

    Returns:
        配置实例

    Examples:
        >>> settings = create_settings()
        >>> settings = create_settings(enable_gpu=True, models_dir="/custom/path")
    """
    return HivisionSettings(**overrides)
