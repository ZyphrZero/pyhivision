#!/usr/bin/env python

"""人脸检测处理器

封装不同检测模型的调用逻辑，提供统一的处理接口。
"""
from pathlib import Path
from typing import Literal

import numpy as np

from core.model_manager import ModelManager
from models.detection.mtcnn import MTCNNModel
from models.detection.retinaface import RetinaFaceModel
from schemas.config import DetectionModelConfig
from schemas.response import FaceInfo
from utils.logger import get_logger

logger = get_logger("processors.detection")

DetectionModelName = Literal["mtcnn", "retinaface"]


class DetectionProcessor:
    """人脸检测处理器"""

    # 模型类注册表（不含文件名，文件名从配置读取）
    _model_class_registry = {
        "mtcnn": MTCNNModel,
        "retinaface": RetinaFaceModel,
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
        model_name: DetectionModelName = "mtcnn",
    ) -> FaceInfo:
        """检测人脸

        Args:
            image: 输入图像 (BGR 格式)
            model_name: 模型名称

        Returns:
            人脸信息

        Raises:
            ValueError: 模型名称不支持或配置中缺少对应的模型文件名
        """
        # 检查模型是否在类注册表中
        if model_name not in self._model_class_registry:
            raise ValueError(f"Unknown detection model: {model_name}")

        # 检查配置中是否有该模型的文件名
        if model_name not in self.model_manager.settings.detection_model_files:
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                f"Please add it to detection_model_files in settings."
            )

        # 从配置读取模型文件名
        model_cls = self._model_class_registry[model_name]
        weight_file = self.model_manager.settings.detection_model_files[model_name]

        # 创建模型配置（从 model_manager 获取配置）
        # weight_file 为 None 表示使用内置权重（如 MTCNN）
        checkpoint_path = (
            self._get_weight_path(weight_file) if weight_file else Path(".")
        )

        config = DetectionModelConfig(
            name=model_name,
            checkpoint_path=checkpoint_path,
            use_gpu=self.model_manager.settings.enable_gpu,
        )

        # 创建模型实例
        model = model_cls(config, self.model_manager)

        # 执行检测
        logger.debug(f"Running detection with model: {model_name}")
        result = model.detect(image)

        return result

    def _get_weight_path(self, filename: str) -> Path:
        """获取权重文件路径

        Args:
            filename: 模型权重文件名

        Returns:
            完整的模型权重文件路径

        Raises:
            ValueError: 如果 detection_models_dir 未配置
        """
        models_dir = self.model_manager.settings.detection_models_dir
        if models_dir is None:
            raise ValueError(
                "检测模型目录未配置。请在创建 SDK 实例时设置 detection_models_dir:\n"
                "  settings = create_settings(detection_models_dir='/path/to/detection/models')\n"
                "  或设置环境变量: HIVISION_DETECTION_MODELS_DIR=/path/to/detection/models\n"
                "  注意：MTCNN 使用内置权重，不需要此配置"
            )
        return models_dir / filename
