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
from pyhivision.processors.model_path_resolver import resolve_model_checkpoint
from pyhivision.schemas.config import DetectionModelConfig
from pyhivision.schemas.response import FaceInfo
from pyhivision.utils.logger import get_logger

logger = get_logger("processors.detection")

DetectionModelName = Literal["mtcnn", "retinaface"]
MTCNN_SCALE = 2


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
        model = self._create_model(model_name)
        logger.debug(f"Running detection with model: {model_name}")
        try:
            return self._run_detection(
                model=model,
                model_name=model_name,
                image=image,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                multiple_faces_strategy=multiple_faces_strategy,
            )
        except ValidationError as e:
            # 转换为业务异常
            error_msg = e.errors()[0]['msg'] if e.errors() else str(e)
            raise FaceDetectionError(
                f"Face detection validation failed: {error_msg}"
            ) from e

    def _create_model(self, model_name: DetectionModelName) -> MTCNNModel | RetinaFaceModel:
        self._validate_model_name(model_name)
        model_cls = self._model_class_registry[model_name]
        config = DetectionModelConfig(
            name=model_name,
            checkpoint_path=self._resolve_checkpoint_path(model_name),
            use_gpu=self.model_manager.settings.enable_gpu,
        )
        return model_cls(config, self.model_manager)

    def _validate_model_name(self, model_name: DetectionModelName) -> None:
        if model_name not in self._model_class_registry:
            raise ValueError(f"Unknown detection model: {model_name}")
        if model_name not in self.model_manager.settings.detection_model_files:
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                "Please add it to detection_model_files in settings."
            )

    def _resolve_checkpoint_path(self, model_name: DetectionModelName) -> Path:
        weight_file = self.model_manager.settings.detection_model_files[model_name]
        if weight_file is None:
            return Path(".")
        return resolve_model_checkpoint(
            settings=self.model_manager.settings,
            model_name=model_name,
            filename=weight_file,
            model_type="detection",
            logger=logger,
        )

    def _run_detection(
        self,
        model: MTCNNModel | RetinaFaceModel,
        model_name: DetectionModelName,
        image: np.ndarray,
        conf_threshold: float,
        nms_threshold: float,
        multiple_faces_strategy: str,
    ) -> FaceInfo:
        """按模型能力执行检测。"""
        if model_name == "retinaface":
            return model.detect(
                image,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                multiple_faces_strategy=multiple_faces_strategy,
            )
        return model.detect(
            image,
            scale=MTCNN_SCALE,
            multiple_faces_strategy=multiple_faces_strategy,
        )
