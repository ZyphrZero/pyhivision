#!/usr/bin/env python

"""抠图处理器

封装不同抠图模型的调用逻辑，提供统一的处理接口。
"""
from typing import Literal

import numpy as np

from pyhivision.core.model_manager import ModelManager
from pyhivision.models.matting.birefnet import BiRefNetModel
from pyhivision.models.matting.modnet import HivisionModNetModel, ModNetPhotographicModel
from pyhivision.models.matting.rmbg import RMBGModel
from pyhivision.processors.model_path_resolver import resolve_model_checkpoint
from pyhivision.schemas.config import MattingModelConfig
from pyhivision.utils.logger import get_logger

logger = get_logger("processors.matting")

MattingModelName = Literal[
    "modnet_photographic",
    "hivision_modnet",
    "birefnet_lite",
    "rmbg_1.4",
    "rmbg_2.0",
]


class MattingProcessor:
    """抠图处理器"""

    # 模型类和参考尺寸映射（不含文件名，文件名从配置读取）
    _model_class_registry = {
        "modnet_photographic": (ModNetPhotographicModel, 512),
        "hivision_modnet": (HivisionModNetModel, 512),
        "birefnet_lite": (BiRefNetModel, 1024),
        "rmbg_1.4": (RMBGModel, 1024),
        "rmbg_2.0": (RMBGModel, 1024),
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
        enable_fix: bool = False,
    ) -> np.ndarray:
        """执行抠图

        Args:
            image: 输入图像 (BGR 格式)
            model_name: 模型名称
            enable_fix: 是否启用抠图修补（仅对 hivision_modnet 有效）

        Returns:
            BGRA 图像（带透明通道）

        Raises:
            ValueError: 模型名称不支持或配置中缺少对应的模型文件名
        """
        model = self._create_model(model_name)

        # 执行推理
        logger.debug(f"Running matting with model: {model_name}")
        result = model.infer(image)

        # 应用修补（可选）
        if enable_fix and model_name in ["hivision_modnet"]:
            result = self._hollow_out_fix(result)

        return result

    def _create_model(
        self,
        model_name: MattingModelName,
    ) -> ModNetPhotographicModel | HivisionModNetModel | BiRefNetModel | RMBGModel:
        self._validate_model_name(model_name)
        model_cls, ref_size = self._model_class_registry[model_name]
        weight_file = self.model_manager.settings.matting_model_files[model_name]
        config = MattingModelConfig(
            name=model_name,
            checkpoint_path=resolve_model_checkpoint(
                settings=self.model_manager.settings,
                model_name=model_name,
                filename=weight_file,
                model_type="matting",
                logger=logger,
            ),
            ref_size=ref_size,
            use_gpu=self.model_manager.settings.enable_gpu,
        )
        return model_cls(config, self.model_manager)

    def _validate_model_name(self, model_name: MattingModelName) -> None:
        if model_name not in self._model_class_registry:
            raise ValueError(f"Unknown matting model: {model_name}")
        if model_name not in self.model_manager.settings.matting_model_files:
            raise ValueError(
                f"Model '{model_name}' not found in configuration. "
                "Please add it to matting_model_files in settings."
            )
        if not self.model_manager.settings.matting_model_files[model_name]:
            raise ValueError(f"Matting model '{model_name}' requires a weight file")

    def _hollow_out_fix(self, src: np.ndarray) -> np.ndarray:
        """修补抠图区域，作为抠图模型精度不够的补充

        Args:
            src: BGRA 图像

        Returns:
            修补后的 BGRA 图像
        """
        import cv2

        b, g, r, a = cv2.split(src)
        src_bgr = cv2.merge((b, g, r))

        # Padding
        add_area = np.zeros((10, a.shape[1]), np.uint8)
        a = np.vstack((add_area, a, add_area))
        add_area = np.zeros((a.shape[0], 10), np.uint8)
        a = np.hstack((add_area, a, add_area))

        # Threshold and erode
        _, a_threshold = cv2.threshold(a, 127, 255, 0)
        a_erode = cv2.erode(
            a_threshold,
            kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=3,
        )

        # Find contours
        contours, hierarchy = cv2.findContours(
            a_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours = [x for x in contours]
        if not contours:
            raise ValueError("Matting fix failed: no foreground contour found in alpha channel")
        contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

        # Draw contour
        a_contour = cv2.drawContours(np.zeros(a.shape, np.uint8), contours[0], -1, 255, 2)

        # Flood fill
        h, w = a.shape[:2]
        mask = np.zeros([h + 2, w + 2], np.uint8)
        cv2.floodFill(a_contour, mask=mask, seedPoint=(0, 0), newVal=255)
        a = cv2.add(a, 255 - a_contour)

        return cv2.merge((src_bgr, a[10:-10, 10:-10]))
