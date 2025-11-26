#!/usr/bin/env python

"""同步处理管线

编排整个证件照处理流程，支持单张和批量处理。
"""

import numpy as np

from pyhivision.config.settings import HivisionSettings
from pyhivision.core.model_manager import ModelManager
from pyhivision.processors.adjustment import AdjustmentProcessor
from pyhivision.processors.alignment import AlignmentProcessor
from pyhivision.processors.background import BackgroundProcessor
from pyhivision.processors.beauty import BeautyProcessor
from pyhivision.processors.detection import DetectionProcessor
from pyhivision.processors.matting import MattingProcessor
from pyhivision.schemas.request import PhotoRequest
from pyhivision.schemas.response import FaceInfo, PhotoResult
from pyhivision.utils.image import parse_color_to_bgr, resize_image_to_max
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

        # 如果启用了下载所有模型选项，则在初始化时下载
        if self.model_manager.settings.download_all_models:
            self._download_all_models()

        # 初始化处理器
        self.matting_processor = MattingProcessor(self.model_manager)
        self.detection_processor = DetectionProcessor(self.model_manager)
        self.alignment_processor = AlignmentProcessor()
        self.adjustment_processor = AdjustmentProcessor(settings=self.model_manager.settings)
        self.beauty_processor = BeautyProcessor(
            lut_image_path=self.model_manager.settings.beauty_lut_path
        )
        self.background_processor = BackgroundProcessor()

        logger.info("PhotoPipeline initialized")

    def _download_all_models(self):
        """下载所有模型"""
        from pyhivision.utils.download import download_all_models, get_default_models_dir

        models_dir = self.model_manager.settings.matting_models_dir
        if models_dir is None:
            models_dir = get_default_models_dir()

        logger.info("开始下载所有模型...")
        download_all_models(models_dir)
        logger.info("所有模型下载完成")

    def _calculate_crop_region(
        self, face_info: "FaceInfo", image_shape: tuple, expand_ratio: float = 3.5
    ) -> tuple[int, int, int, int]:
        """根据人脸信息计算裁剪区域

        Args:
            face_info: 人脸信息
            image_shape: 图像尺寸 (height, width)
            expand_ratio: 扩展比例（人脸宽度的倍数，默认3.5以包含头部到胸部）

        Returns:
            (x1, y1, x2, y2): 裁剪区域坐标
        """
        h, w = image_shape[:2]

        # 人脸中心
        face_center_x = face_info.x + face_info.width / 2
        face_center_y = face_info.y + face_info.height / 2

        # 扩展区域（证件照需要包含头部到胸部）
        crop_size = max(face_info.width, face_info.height) * expand_ratio

        # 计算裁剪框（保持正方形或接近正方形）
        x1 = max(0, int(face_center_x - crop_size / 2))
        y1 = max(0, int(face_center_y - crop_size / 2))
        x2 = min(w, int(face_center_x + crop_size / 2))
        y2 = min(h, int(face_center_y + crop_size / 2))

        return x1, y1, x2, y2

    def process_single(self, request: PhotoRequest) -> PhotoResult:
        """处理单张照片

        1. 全图抠图
        2. 检测人脸（在抠图结果上）
        3. 美颜处理
        4. 人脸矫正（如果需要）
        5. 重新检测（如果做了矫正）
        6. 图像调整（裁剪）
        7. 背景替换

        Args:
            request: 处理请求

        Returns:
            处理结果
        """
        import cv2

        # 预处理：限制图像大小
        image = request.image
        if max(image.shape[:2]) > self.model_manager.settings.max_image_size:
            image = resize_image_to_max(image, self.model_manager.settings.max_image_size)

        # 1. 全图抠图
        logger.debug("Step 1: Matting on full image")
        matting_result = self.matting_processor.process(
            image, request.matting_model, request.enable_matting_fix
        )

        # 2. 人脸检测（在抠图结果的RGB通道上检测）
        face_info = None
        if not request.change_bg_only:
            logger.debug("Step 2: Face detection on matting result")
            # 分离RGB通道用于检测
            b, g, r, a = cv2.split(matting_result)
            rgb_for_detection = cv2.merge((b, g, r))

            face_info = self.detection_processor.process(
                rgb_for_detection,
                request.detection_model,
                conf_threshold=request.detection_confidence_threshold,
                nms_threshold=request.detection_nms_threshold,
                multiple_faces_strategy=request.multiple_faces_strategy,
            )
            logger.debug(
                f"Face detected: position=({face_info.x}, {face_info.y}), "
                f"size=({face_info.width}x{face_info.height}), "
                f"roll_angle={face_info.roll_angle:.2f}°"
            )

        # 3. 美颜处理
        logger.debug("Step 3: Beauty processing")
        beautified = self.beauty_processor.process(matting_result, request.beauty_params)

        # 4. 人脸矫正（如果需要）
        if (
            face_info
            and request.alignment_params.enable_alignment
            and abs(face_info.roll_angle) > request.alignment_params.angle_threshold
        ):
            logger.info(
                f"Step 4: Face alignment triggered - roll_angle={face_info.roll_angle:.2f}°, "
                f"threshold={request.alignment_params.angle_threshold:.2f}°"
            )

            # 分离 BGRA 通道
            b, g, r, a = cv2.split(beautified)
            rgb_image = cv2.merge((b, g, r))

            # 执行矫正
            alignment_result = self.alignment_processor.process(
                image=rgb_image,
                alpha=a,
                roll_angle=face_info.roll_angle,
                params=request.alignment_params,
            )

            # 合并通道
            beautified = cv2.merge((
                *cv2.split(alignment_result.rotated_image),
                alignment_result.rotated_alpha,
            ))

            # 5. 变换人脸坐标（使用旋转矩阵，避免重新检测）
            logger.debug("Step 5: Transforming face coordinates after alignment...")
            face_info = alignment_result.transform_face_info(face_info)
            logger.info(
                f"Face alignment complete - roll_angle corrected to {face_info.roll_angle:.2f}°"
            )
        else:
            if face_info:
                logger.debug(
                    f"Step 4: Skip alignment - roll_angle={face_info.roll_angle:.2f}° "
                    f"<= threshold={request.alignment_params.angle_threshold:.2f}°"
                )

        # 6. 图像调整（裁剪、缩放、布局）
        logger.debug("Step 6: Image adjustment")
        if face_info:
            standard, hd = self.adjustment_processor.process(
                beautified, face_info, request.size, request.layout_params
            )
        else:
            # 换底模式：直接缩放到目标尺寸
            h, w = request.size
            standard = cv2.resize(beautified, (w, h), interpolation=cv2.INTER_AREA)
            hd = beautified.copy()

        # 7. 背景替换（可选）
        logger.debug("Step 7: Background replacement")
        if request.add_background:
            # 智能转换颜色格式为 BGR（OpenCV 格式）
            bg_color = parse_color_to_bgr(request.background_color, request.color_format)

            # 添加背景色：BGRA → BGR
            standard_result = self.background_processor.add_background(
                standard, bg_color
            )
            hd_result = (
                self.background_processor.add_background(hd, bg_color)
                if request.render_hd
                else None
            )
        else:
            # 保持透明背景：BGRA 格式
            standard_result = standard
            hd_result = hd if request.render_hd else None

        # 返回结果
        return PhotoResult(
            standard=standard_result,
            hd=hd_result,
            matting=matting_result if request.render_matting else None,
            face_info=face_info,
        )

    def process_batch(
        self,
        requests: list[PhotoRequest],
        max_workers: int = 4,
    ) -> list[PhotoResult | Exception]:
        """批量处理照片（使用多线程）

        Args:
            requests: 处理请求列表
            max_workers: 最大线程数

        Returns:
            处理结果列表（失败的请求返回 Exception）
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(f"Processing batch of {len(requests)} photos with {max_workers} workers")

        results = [None] * len(requests)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.process_single, req): i
                for i, req in enumerate(requests)
            }

            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Failed to process photo at index {index}: {e}", exc_info=True)
                    results[index] = e

        return results

    def shutdown(self):
        """关闭管线，清理资源"""
        logger.info("Shutting down PhotoPipeline...")
        self.model_manager.shutdown()
        logger.info("PhotoPipeline shutdown complete")

    def __enter__(self):
        """进入上下文"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，自动清理资源"""
        self.shutdown()
        return False  # 不抑制异常


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
