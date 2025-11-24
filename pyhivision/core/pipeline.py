#!/usr/bin/env python

"""同步处理管线

编排整个证件照处理流程，支持单张和批量处理。
"""

import numpy as np

from pyhivision.config.settings import HivisionSettings
from pyhivision.core.model_manager import ModelManager
from pyhivision.processors.adjustment import AdjustmentProcessor
from pyhivision.processors.background import BackgroundProcessor
from pyhivision.processors.beauty import BeautyProcessor
from pyhivision.processors.detection import DetectionProcessor
from pyhivision.processors.matting import MattingProcessor
from pyhivision.schemas.request import PhotoRequest
from pyhivision.schemas.response import PhotoResult
from pyhivision.utils.image import resize_image_to_max, rotate_image_4channels
from pyhivision.utils.logger import get_logger

logger = get_logger("core.pipeline")


class PhotoPipeline:
    """同步证件照处理管线"""

    def __init__(self, model_manager: ModelManager | None = None):
        """初始化管线

        Args:
            model_manager: 模型管理器（可选）
        """
        self.model_manager = model_manager or ModelManager()

        # 初始化处理器
        self.matting_processor = MattingProcessor(self.model_manager)
        self.detection_processor = DetectionProcessor(self.model_manager)
        self.adjustment_processor = AdjustmentProcessor(settings=self.model_manager.settings)
        self.beauty_processor = BeautyProcessor(
            lut_image_path=self.model_manager.settings.beauty_lut_path
        )
        self.background_processor = BackgroundProcessor()

        logger.info("PhotoPipeline initialized")

    def process_single(self, request: PhotoRequest) -> PhotoResult:
        """处理单张照片

        Args:
            request: 处理请求

        Returns:
            处理结果
        """
        # 预处理：限制图像大小
        image = request.image
        if max(image.shape[:2]) > self.model_manager.settings.max_image_size:
            image = resize_image_to_max(image, self.model_manager.settings.max_image_size)

        # 1. 人像抠图
        matting_result = self.matting_processor.process(image, request.matting_model)

        # 2. 人脸检测（仅在非换底模式下执行）
        if not request.change_bg_only:
            face_info = self.detection_processor.process(image, request.detection_model)
        else:
            face_info = None

        # 3. 美颜处理
        beautified = self.beauty_processor.process(matting_result, request.beauty_params)

        # 4. 人脸矫正（如果需要且有人脸信息）
        if face_info and face_info.angle != 0 and request.face_alignment:
            beautified = rotate_image_4channels(beautified, -face_info.angle)

        # 5. 图像调整（裁剪、缩放、布局）
        if face_info:
            standard, hd = self.adjustment_processor.process(
                beautified, face_info, request.size, request.layout_params
            )
        else:
            # 换底模式：直接缩放到目标尺寸
            import cv2
            h, w = request.size
            standard = cv2.resize(beautified, (w, h), interpolation=cv2.INTER_AREA)
            hd = beautified.copy()

        # 6. 背景替换
        result = self.background_processor.add_background(standard, request.background_color)

        # 7. 返回结果
        return PhotoResult(
            standard=result,
            hd=hd if request.render_hd else None,
            matting=matting_result if request.render_matting else None,
            face_info=face_info,
        )

    def process_batch(
        self,
        requests: list[PhotoRequest],
        max_workers: int = 4,
    ) -> list[PhotoResult | Exception]:
        """批量处理照片（使用多进程）

        Args:
            requests: 处理请求列表
            max_workers: 最大进程数

        Returns:
            处理结果列表（失败的请求返回 Exception）
        """
        logger.info(f"Processing batch of {len(requests)} photos with {max_workers} workers")

        results = []
        for request in requests:
            try:
                result = self.process_single(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process photo: {e}", exc_info=True)
                results.append(e)

        return results

    def shutdown(self):
        """关闭管线，清理资源"""
        logger.info("Shutting down PhotoPipeline...")
        self.model_manager.shutdown()
        logger.info("PhotoPipeline shutdown complete")


class IDPhotoSDK:
    """证件照 SDK 主入口"""

    @staticmethod
    def create(settings: HivisionSettings | None = None) -> PhotoPipeline:
        """创建处理管线

        Args:
            settings: 配置实例（可选）

        Returns:
            PhotoPipeline 实例
        """
        model_manager = ModelManager(settings)
        return PhotoPipeline(model_manager)

    @staticmethod
    def process(
        image: np.ndarray,
        size: tuple[int, int],
        background_color: tuple[int, int, int] = (255, 255, 255),
        matting_model: str = "modnet_photographic",
        detection_model: str = "mtcnn",
        settings: HivisionSettings | None = None,
    ) -> PhotoResult:
        """快速处理接口（单次使用）

        Args:
            image: 输入图像 (BGR 格式)
            size: 目标尺寸 (高度, 宽度)
            background_color: 背景颜色 (BGR)
            matting_model: 抠图模型名称
            detection_model: 检测模型名称
            settings: 配置实例（可选）

        Returns:
            处理结果
        """
        pipeline = IDPhotoSDK.create(settings)
        request = PhotoRequest(
            image=image,
            size=size,
            background_color=background_color,
            matting_model=matting_model,
            detection_model=detection_model,
        )
        result = pipeline.process_single(request)
        pipeline.shutdown()
        return result
