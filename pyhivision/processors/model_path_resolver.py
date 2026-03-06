#!/usr/bin/env python

"""模型权重路径解析工具。"""

from pathlib import Path
from typing import Literal

from pyhivision.config.settings import HivisionSettings
from pyhivision.utils.download import download_model, get_default_models_dir
from pyhivision.utils.logger import LoggerProtocol

ModelType = Literal["matting", "detection"]

MODEL_HINTS = (
    "💡 推荐方式（最简单）：\n"
    "   在命令行运行：\n"
    "   $ pyhivision install {model_name}\n\n"
    "📦 其他方式：\n"
    "   1. 在代码中下载：\n"
    "      from pyhivision import download_model\n"
    "      download_model('{model_name}', '{model_type}')\n\n"
    "   2. 启用自动下载：\n"
    "      settings = create_settings(auto_download_models=True)\n\n"
    "   3. 下载所有模型：\n"
    "      $ pyhivision install --all\n"
)


def resolve_model_checkpoint(
    settings: HivisionSettings,
    model_name: str,
    filename: str,
    model_type: ModelType,
    logger: LoggerProtocol,
) -> Path:
    """解析模型权重路径，不存在时按配置触发自动下载。"""
    models_dir = _resolve_models_dir(settings, model_type)
    model_path = models_dir / filename
    if model_path.exists():
        return model_path

    if settings.auto_download_models:
        logger.info(f"模型文件不存在，自动下载: {filename}")
        return download_model(model_name, model_type, models_dir.parent)

    raise FileNotFoundError(_format_missing_model_error(model_name, model_type, model_path.name))


def _resolve_models_dir(settings: HivisionSettings, model_type: ModelType) -> Path:
    if model_type == "matting":
        configured_dir = settings.matting_models_dir
    else:
        configured_dir = settings.detection_models_dir

    if configured_dir is not None:
        return Path(configured_dir)
    return get_default_models_dir() / model_type


def _format_missing_model_error(model_name: str, model_type: ModelType, filename: str) -> str:
    hints = MODEL_HINTS.format(model_name=model_name, model_type=model_type)
    return (
        f"\n{'=' * 60}\n"
        f"❌ 模型文件不存在: {filename}\n"
        f"{'=' * 60}\n\n"
        f"{hints}"
        f"{'=' * 60}\n"
    )

