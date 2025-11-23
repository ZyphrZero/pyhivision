#!/usr/bin/env python

"""异步处理管线

编排整个证件照处理流程，支持单张和批量处理。
"""
import asyncio
import time

import numpy as np

from config.settings import HivisionSettings
from core.metrics import MetricsCollector
from core.model_manager import ModelManager
from processors.adjustment import AdjustmentProcessor
from processors.background import BackgroundProcessor
from processors.beauty import BeautyProcessor
from processors.detection import DetectionProcessor
from processors.matting import MattingProcessor
from schemas.request import PhotoRequest
from schemas.response import PhotoResult
from utils.image import resize_image_to_max, rotate_image_4channels
from utils.logger import get_logger

logger = get_logger("core.pipeline")


class AsyncPhotoPipeline:
    """异步证件照处理管线"""

    def __init__(self, model_manager: ModelManager | None = None):
        """初始化管线

        Args:
            model_manager: 模型管理器（可选，默认使用全局单例）
        """
        self.model_manager = model_manager or ModelManager()

        # 初始化处理器（传递配置）
        self.matting_processor = MattingProcessor(self.model_manager)
        self.detection_processor = DetectionProcessor(self.model_manager)
        self.adjustment_processor = AdjustmentProcessor(settings=self.model_manager.settings)
        self.beauty_processor = BeautyProcessor(
            lut_image_path=self.model_manager.settings.beauty_lut_path
        )
        self.background_processor = BackgroundProcessor()

        logger.info("AsyncPhotoPipeline initialized")

    async def _safe_call_hook(
        self,
        hook_name: str,
        hooks: dict,
        context: dict,
        timeout_seconds: float = 10.0,
    ) -> dict | None:
        """安全调用钩子函数

        Args:
            hook_name: 钩子名称
            hooks: 钩子函数字典
            context: 上下文数据
            timeout_seconds: 超时时间（秒）

        Returns:
            钩子返回值（失败时返回 None）
        """
        if hook_name not in hooks:
            return None

        try:
            hook_result = await asyncio.wait_for(
                hooks[hook_name](context), timeout=timeout_seconds
            )
            return hook_result
        except TimeoutError:
            logger.error(f"Hook '{hook_name}' timed out after {timeout_seconds}s")
            return None
        except Exception as e:
            logger.error(f"Hook '{hook_name}' failed: {e}", exc_info=True)
            return None

    async def process_single(self, request: PhotoRequest) -> PhotoResult:
        """处理单张照片

        Args:
            request: 处理请求

        Returns:
            处理结果
        """
        metrics = MetricsCollector()
        start_time = time.time()

        # 预处理：限制图像大小
        image = request.image
        if max(image.shape[:2]) > self.model_manager.settings.max_image_size:
            image = resize_image_to_max(image, self.model_manager.settings.max_image_size)

        # 1. 人像抠图
        async with metrics.measure("matting"):
            matting_result = await self.matting_processor.process(
                image, request.matting_model
            )

        # 调用 after_matting 钩子
        hook_result = await self._safe_call_hook(
            "after_matting",
            request.hooks,
            {
                "matting": matting_result,
                "original_image": image,
            },
        )
        if hook_result and "matting" in hook_result:
            matting_result = hook_result["matting"]

        # 2. 并行执行美颜和人脸检测
        async with metrics.measure("parallel_processing"):
            # 美颜处理
            beauty_task = asyncio.create_task(
                self.beauty_processor.process(matting_result, request.beauty_params)
            )

            # 人脸检测（仅在非换底模式下执行）
            if not request.change_bg_only:
                detection_task = asyncio.create_task(
                    self.detection_processor.process(image, request.detection_model)
                )
                beautified, face_info = await asyncio.gather(beauty_task, detection_task)
            else:
                beautified = await beauty_task
                face_info = None

        # 调用 after_detection 钩子
        hook_result = await self._safe_call_hook(
            "after_detection",
            request.hooks,
            {
                "face_info": face_info,
                "matting": matting_result,
            },
        )
        if hook_result and "face_info" in hook_result:
            face_info = hook_result["face_info"]

        # 调用 after_beauty 钩子
        hook_result = await self._safe_call_hook(
            "after_beauty",
            request.hooks,
            {
                "beautified": beautified,
                "face_info": face_info,
            },
        )
        if hook_result and "beautified" in hook_result:
            beautified = hook_result["beautified"]

        # 2.5 人脸矫正（如果启用且角度超过阈值）
        if (
            face_info is not None
            and request.face_alignment
            and abs(face_info.roll_angle) > 2
        ):
            async with metrics.measure("face_alignment"):
                logger.debug(f"Face alignment needed, roll_angle={face_info.roll_angle:.2f}°")

                # 分离 BGRA 通道
                import cv2
                b, g, r, a = cv2.split(beautified)

                # 4 通道旋转
                rotated_bgr, beautified, cos, sin, d_w, d_h = rotate_image_4channels(
                    cv2.merge((b, g, r)),
                    a,
                    -face_info.roll_angle,
                )

                # 重新检测人脸（基于旋转后的图像）
                face_info = await self.detection_processor.process(
                    rotated_bgr, request.detection_model,
                )

        # 3. 图像调整
        if face_info is not None:
            async with metrics.measure("adjustment"):
                standard, hd = await self.adjustment_processor.process(
                    beautified,
                    face_info,
                    request.size,
                    request.layout_params,
                )
        else:
            # 仅换底模式：直接调整大小
            import cv2

            standard = cv2.resize(
                beautified, (request.size[1], request.size[0]), interpolation=cv2.INTER_AREA
            )
            hd = beautified if request.render_hd else None

        # 调用 after_adjustment 钩子
        hook_result = await self._safe_call_hook(
            "after_adjustment",
            request.hooks,
            {
                "standard": standard,
                "hd": hd,
                "face_info": face_info,
            },
        )
        if hook_result:
            if "standard" in hook_result:
                standard = hook_result["standard"]
            if "hd" in hook_result and hd is not None:
                hd = hook_result["hd"]

        # 4. 添加背景
        async with metrics.measure("background"):
            standard_with_bg = await self.background_processor.add_background(
                standard, request.background_color
            )
            if hd is not None:
                hd_with_bg = await self.background_processor.add_background(
                    hd, request.background_color
                )
            else:
                hd_with_bg = None

        # 调用 after_background 钩子
        hook_result = await self._safe_call_hook(
            "after_background",
            request.hooks,
            {
                "standard": standard_with_bg,
                "hd": hd_with_bg,
                "matting": matting_result,
                "face_info": face_info,
            },
        )
        if hook_result:
            if "standard" in hook_result:
                standard_with_bg = hook_result["standard"]
            if "hd" in hook_result and hd_with_bg is not None:
                hd_with_bg = hook_result["hd"]

        # 计算总时间
        total_time_ms = (time.time() - start_time) * 1000
        stage_times = metrics.get_current_request_times()

        logger.info(f"Processing completed in {total_time_ms:.2f}ms")

        return PhotoResult(
            standard=standard_with_bg,
            hd=hd_with_bg,
            matting=matting_result,
            face_info=face_info,
            processing_time_ms=total_time_ms,
            stage_times=stage_times,
            model_info={
                "matting": request.matting_model,
                "detection": request.detection_model if not request.change_bg_only else "none",
            },
        )

    async def process_batch(
        self,
        requests: list[PhotoRequest],
        batch_size: int = 4,
        fail_fast: bool = False,
    ) -> list[PhotoResult | Exception]:
        """批量处理照片

        Args:
            requests: 请求列表
            batch_size: 每批大小
            fail_fast: 是否在遇到第一个错误时立即停止（默认 False，继续处理）

        Returns:
            结果列表（可能包含 Exception 对象）

        Note:
            当 fail_fast=False 时，失败的请求会在结果列表中以 Exception 对象形式保留
            当 fail_fast=True 时，遇到第一个错误会立即抛出异常并停止处理
        """
        results = []

        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]

            # 并发处理每批
            batch_results = await asyncio.gather(
                *[self.process_single(req) for req in batch],
                return_exceptions=True,
            )

            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Request {i + idx} failed: {result}")
                    if fail_fast:
                        raise result
                    results.append(result)  # 保留异常对象
                else:
                    results.append(result)

        return results


class IDPhotoSDK:
    """IDPhoto SDK 主入口类

    提供便捷的创建和使用接口。
    """

    @staticmethod
    def create(settings: HivisionSettings | None = None) -> AsyncPhotoPipeline:
        """创建处理管线实例

        Args:
            settings: 配置实例（可选）
        """
        model_manager = ModelManager(settings=settings)
        return AsyncPhotoPipeline(model_manager=model_manager)

    @staticmethod
    async def process(
        image: np.ndarray,
        size: tuple[int, int] = (413, 295),
        background_color: tuple[int, int, int] = (255, 255, 255),
        matting_model: str = "modnet_photographic",
        detection_model: str = "mtcnn",
        **kwargs,
    ) -> PhotoResult:
        """快速处理接口

        Args:
            image: 输入图像
            size: 目标尺寸
            background_color: 背景颜色
            matting_model: 抠图模型
            detection_model: 检测模型
            **kwargs: 其他参数

        Returns:
            处理结果
        """
        pipeline = AsyncPhotoPipeline()

        request = PhotoRequest(
            image=image,
            size=size,
            background_color=background_color,
            matting_model=matting_model,
            detection_model=detection_model,
            **kwargs,
        )

        return await pipeline.process_single(request)
