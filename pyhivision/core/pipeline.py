#!/usr/bin/env python

"""同步处理管线

编排整个证件照处理流程，支持单张和批量处理。
"""

import cv2
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

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        """预处理输入图像尺寸。"""
        max_size = self.model_manager.settings.max_image_size
        if max(image.shape[:2]) > max_size:
            return resize_image_to_max(image, max_size)
        return image

    def _detect_face(
        self,
        matting_result: np.ndarray,
        request: PhotoRequest,
    ) -> FaceInfo | None:
        """在抠图结果上做人脸检测。"""
        if request.change_bg_only:
            return None

        logger.debug("Step 2: Face detection on matting result")
        rgb_for_detection = matting_result[:, :, :3].copy()
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
        return face_info

    def _align_face_if_needed(
        self,
        image: np.ndarray,
        face_info: FaceInfo | None,
        request: PhotoRequest,
    ) -> tuple[np.ndarray, FaceInfo | None]:
        """按需执行人脸矫正，并同步变换人脸框。"""
        if face_info is None:
            return image, face_info

        threshold = request.alignment_params.angle_threshold
        if not request.alignment_params.enable_alignment or abs(face_info.roll_angle) <= threshold:
            logger.debug(
                f"Step 4: Skip alignment - roll_angle={face_info.roll_angle:.2f}° "
                f"<= threshold={threshold:.2f}°"
            )
            return image, face_info

        logger.info(
            f"Step 4: Face alignment triggered - roll_angle={face_info.roll_angle:.2f}°, "
            f"threshold={threshold:.2f}°"
        )
        rgb_image = image[:, :, :3]
        alpha = image[:, :, 3]
        alignment_result = self.alignment_processor.process(
            image=rgb_image,
            alpha=alpha,
            roll_angle=face_info.roll_angle,
            params=request.alignment_params,
        )
        if alignment_result.rotated_alpha is None:
            raise ValueError("Alignment result missing alpha channel")

        aligned = np.dstack([alignment_result.rotated_image, alignment_result.rotated_alpha])
        logger.debug("Step 5: Transforming face coordinates after alignment...")
        transformed_face_info = alignment_result.transform_face_info(face_info)
        logger.info(
            f"Face alignment complete - roll_angle corrected to {transformed_face_info.roll_angle:.2f}°"
        )
        return aligned, transformed_face_info

    def _adjust_outputs(
        self,
        image: np.ndarray,
        face_info: FaceInfo | None,
        request: PhotoRequest,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """生成人像标准照/高清照。"""
        logger.debug("Step 6: Image adjustment")
        if face_info is None:
            target_h, target_w = request.size
            standard = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            hd = image.copy() if request.render_hd else None
            return standard, hd

        return self.adjustment_processor.process(
            image=image,
            face_info=face_info,
            target_size=request.size,
            layout_params=request.layout_params,
            render_hd=request.render_hd,
        )

    def _apply_background(
        self,
        standard: np.ndarray,
        hd: np.ndarray | None,
        request: PhotoRequest,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """按需执行背景替换。"""
        logger.debug("Step 7: Background replacement")
        if not request.add_background:
            return standard, hd

        bg_color = parse_color_to_bgr(request.background_color)
        standard_result = self.background_processor.add_background(standard, bg_color)
        hd_result = self.background_processor.add_background(hd, bg_color) if hd is not None else None
        return standard_result, hd_result

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
        image = self._prepare_input(request.image)
        logger.debug("Step 1: Matting on full image")
        matting_result = self.matting_processor.process(
            image, request.matting_model, request.enable_matting_fix
        )
        face_info = self._detect_face(matting_result, request)
        logger.debug("Step 3: Beauty processing")
        beautified = self.beauty_processor.process(matting_result, request.beauty_params)
        beautified, face_info = self._align_face_if_needed(beautified, face_info, request)
        standard, hd = self._adjust_outputs(beautified, face_info, request)
        standard_result, hd_result = self._apply_background(standard, hd, request)

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
        background_color: tuple[int, int, int] | str = (255, 255, 255),
        matting_model: str = "modnet_photographic",
        detection_model: str = "mtcnn",
        settings: HivisionSettings | None = None,
    ) -> PhotoResult:
        """快速处理接口（单次使用）

        Args:
            image: 输入图像 (BGR 格式)
            size: 目标尺寸 (高度, 宽度)
            background_color: 背景颜色（RGB 元组或十六进制字符串）
            matting_model: 抠图模型名称
            detection_model: 检测模型名称
            settings: 配置实例（可选）

        Returns:
            处理结果
        """
        request = PhotoRequest(
            image=image,
            size=size,
            background_color=background_color,
            matting_model=matting_model,
            detection_model=detection_model,
        )
        with IDPhotoSDK.create(settings) as pipeline:
            return pipeline.process_single(request)
