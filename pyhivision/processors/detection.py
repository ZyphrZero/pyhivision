#!/usr/bin/env python

"""人脸检测处理器

封装不同检测模型的调用逻辑，提供统一的处理接口。
"""
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ValidationError

from pyhivision.core.model_manager import ModelManager
from pyhivision.exceptions.errors import FaceDetectionError
from pyhivision.models.detection.mtcnn import MTCNNModel
from pyhivision.models.detection.retinaface import RetinaFaceModel
from pyhivision.schemas.config import DetectionModelConfig
from pyhivision.schemas.response import FaceInfo
from pyhivision.utils.logger import get_logger

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
        conf_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        multiple_faces_strategy: str = "best",
    ) -> FaceInfo:
        """检测人脸

        Args:
            image: 输入图像 (BGR 格式)
            model_name: 模型名称
            conf_threshold: 置信度阈值（默认 0.8）
            nms_threshold: NMS IoU 阈值（默认 0.3）
            multiple_faces_strategy: 多人脸处理策略（默认 "best"）
                - "error": 检测到多人脸时报错（严格模式）
                - "best": 选择置信度最高的人脸
                - "largest": 选择面积最大的人脸

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

        # 执行检测（根据模型类型传递不同参数）
        logger.debug(f"Running detection with model: {model_name}")
        try:
            if model_name == "retinaface":
                # RetinaFace 支持完整的 NMS 配置
                result = model.detect(
                    image,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold,
                    multiple_faces_strategy=multiple_faces_strategy,
                )
            else:  # mtcnn
                # MTCNN 只支持多人脸策略（内置 NMS）
                result = model.detect(
                    image,
                    scale=2,
                    multiple_faces_strategy=multiple_faces_strategy,
                )
            return result
        except ValidationError as e:
            # 转换为业务异常
            error_msg = e.errors()[0]['msg'] if e.errors() else str(e)
            raise FaceDetectionError(
                f"Face detection validation failed: {error_msg}"
            ) from e

    def _get_weight_path(self, filename: str) -> Path:
        """获取权重文件路径，如果不存在则提示或自动下载

        Args:
            filename: 模型权重文件名

        Returns:
            完整的模型权重文件路径

        Raises:
            ValueError: 如果 detection_models_dir 未配置
            FileNotFoundError: 如果模型文件不存在且未启用自动下载
        """
        from pyhivision.utils.download import download_model, get_default_models_dir

        models_dir = self.model_manager.settings.detection_models_dir
        if models_dir is None:
            models_dir = get_default_models_dir() / "detection"

        model_path = models_dir / filename

        # 检查文件是否存在
        if not model_path.exists():
            model_name = next((k for k, v in self.model_manager.settings.detection_model_files.items() if v == filename), None)

            if self.model_manager.settings.auto_download_models:
                logger.info(f"模型文件不存在，自动下载: {filename}")
                return download_model(model_name, "detection", models_dir.parent)
            else:
                raise FileNotFoundError(
                    f"\n{'='*60}\n"
                    f"❌ 模型文件不存在: {model_path.name}\n"
                    f"{'='*60}\n\n"
                    f"💡 推荐方式（最简单）：\n"
                    f"   在命令行运行：\n"
                    f"   $ pyhivision install {model_name}\n\n"
                    f"📦 其他方式：\n"
                    f"   1. 在代码中下载：\n"
                    f"      from pyhivision import download_model\n"
                    f"      download_model('{model_name}', 'detection')\n\n"
                    f"   2. 启用自动下载：\n"
                    f"      settings = create_settings(auto_download_models=True)\n\n"
                    f"   3. 下载所有模型：\n"
                    f"      $ pyhivision install --all\n"
                    f"{'='*60}\n"
                )

        return model_path
