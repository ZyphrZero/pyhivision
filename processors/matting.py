#!/usr/bin/env python

"""抠图处理器

封装不同抠图模型的调用逻辑，提供统一的处理接口。
"""
from pathlib import Path
from typing import Literal

import numpy as np

from core.model_manager import ModelManager
from models.matting.birefnet import BiRefNetModel
from models.matting.modnet import HivisionModNetModel, ModNetPhotographicModel
from models.matting.rmbg import RMBGModel
from schemas.config import MattingModelConfig
from utils.logger import get_logger

logger = get_logger("processors.matting")

MattingModelName = Literal[
    "modnet_photographic",
    "hivision_modnet",
    "birefnet_lite",
    "rmbg_1_4",
]


class MattingProcessor:
    """抠图处理器"""

    # 模型类和参考尺寸映射（不含文件名，文件名从配置读取）
    _model_class_registry = {
        "modnet_photographic": (ModNetPhotographicModel, 512),
        "hivision_modnet": (HivisionModNetModel, 512),
        "birefnet_lite": (BiRefNetModel, 1024),
        "rmbg_1_4": (RMBGModel, 1024),
    }

    def __init__(self, model_manager: ModelManager):
        """初始化处理器

        Args:
            model_manager: 模型管理器实例
        """
        self.model_manager = model_manager

    def process(
        self,
        image: np.ndarray,
        model_name: MattingModelName = "modnet_photographic",
    ) -> np.ndarray:
        """执行抠图

        Args:
            image: 输入图像 (BGR 格式)
            model_name: 模型名称

        Returns:
            BGRA 图像（带透明通道）

        Raises:
            ValueError: 模型名称不支持或配置中缺少对应的模型文件名
        """
        # 检查模型是否在类注册表中
        if model_name not in self._model_class_registry:
            raise ValueError(f"Unknown matting model: {model_name}")

        # 检查配置中是否有该模型的文件名
        if model_name not in self.model_manager.settings.matting_model_files:
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                f"Please add it to matting_model_files in settings."
            )

        # 从配置读取模型文件名
        model_cls, ref_size = self._model_class_registry[model_name]
        weight_file = self.model_manager.settings.matting_model_files[model_name]

        # 创建模型配置（从 model_manager 获取配置）
        config = MattingModelConfig(
            name=model_name,
            checkpoint_path=self._get_weight_path(weight_file),
            ref_size=ref_size,
            use_gpu=self.model_manager.settings.enable_gpu,
        )

        # 创建模型实例
        model = model_cls(config, self.model_manager)

        # 执行推理
        logger.debug(f"Running matting with model: {model_name}")
        result = model.infer(image)

        return result

    def _get_weight_path(self, filename: str) -> Path:
        """获取权重文件路径

        Args:
            filename: 模型权重文件名

        Returns:
            完整的模型权重文件路径

        Raises:
            ValueError: 如果 matting_models_dir 未配置
        """
        models_dir = self.model_manager.settings.matting_models_dir
        if models_dir is None:
            raise ValueError(
                "抠图模型目录未配置。请在创建 SDK 实例时设置 matting_models_dir:\n"
                "  settings = create_settings(matting_models_dir='/path/to/matting/models')\n"
                "  或设置环境变量: HIVISION_MATTING_MODELS_DIR=/path/to/matting/models"
            )
        return models_dir / filename
