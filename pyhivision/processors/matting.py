#!/usr/bin/env python

"""抠图处理器

封装不同抠图模型的调用逻辑，提供统一的处理接口。
"""
from pathlib import Path
from typing import Literal

import numpy as np

from pyhivision.core.model_manager import ModelManager
from pyhivision.models.matting.birefnet import BiRefNetModel
from pyhivision.models.matting.modnet import HivisionModNetModel, ModNetPhotographicModel
from pyhivision.models.matting.rmbg import RMBGModel
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

        # 应用修补（可选）
        if enable_fix and model_name in ["hivision_modnet"]:
            result = self._hollow_out_fix(result)

        return result

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
        contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

        # Draw contour
        a_contour = cv2.drawContours(np.zeros(a.shape, np.uint8), contours[0], -1, 255, 2)

        # Flood fill
        h, w = a.shape[:2]
        mask = np.zeros([h + 2, w + 2], np.uint8)
        cv2.floodFill(a_contour, mask=mask, seedPoint=(0, 0), newVal=255)
        a = cv2.add(a, 255 - a_contour)

        return cv2.merge((src_bgr, a[10:-10, 10:-10]))

    def _get_weight_path(self, filename: str) -> Path:
        """获取权重文件路径，如果不存在则提示或自动下载

        Args:
            filename: 模型权重文件名

        Returns:
            完整的模型权重文件路径

        Raises:
            ValueError: 如果 matting_models_dir 未配置
            FileNotFoundError: 如果模型文件不存在且未启用自动下载
        """
        from pyhivision.utils.download import download_model, get_default_models_dir

        models_dir = self.model_manager.settings.matting_models_dir
        if models_dir is None:
            models_dir = get_default_models_dir() / "matting"

        model_path = models_dir / filename

        # 检查文件是否存在
        if not model_path.exists():
            model_name = next((k for k, v in self.model_manager.settings.matting_model_files.items() if v == filename), None)

            if self.model_manager.settings.auto_download_models:
                logger.info(f"模型文件不存在，自动下载: {filename}")
                return download_model(model_name, "matting", models_dir.parent)
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
                    f"      download_model('{model_name}', 'matting')\n\n"
                    f"   2. 启用自动下载：\n"
                    f"      settings = create_settings(auto_download_models=True)\n\n"
                    f"   3. 下载所有模型：\n"
                    f"      $ pyhivision install --all\n"
                    f"{'='*60}\n"
                )

        return model_path
