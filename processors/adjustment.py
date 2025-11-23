#!/usr/bin/env python

"""图像调整处理器

负责 ID Photo 的裁剪、缩放和布局调整。
"""
import cv2
import numpy as np

from config.settings import HivisionSettings, create_settings
from schemas.request import LayoutParams
from schemas.response import FaceInfo


class AdjustmentProcessor:
    """图像调整处理器"""

    def __init__(self, settings: HivisionSettings | None = None):
        """初始化处理器

        Args:
            settings: 配置实例（可选）
        """
        self._settings = settings

    @property
    def settings(self) -> HivisionSettings:
        """获取配置（延迟初始化）"""
        if self._settings is None:
            self._settings = create_settings()
        return self._settings

    async def process(
        self,
        image: np.ndarray,
        face_info: FaceInfo,
        target_size: tuple[int, int],
        layout_params: LayoutParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """调整图像

        Args:
            image: 输入图像 (BGRA 格式)
            face_info: 人脸信息
            target_size: 目标尺寸 (高度, 宽度)
            layout_params: 布局参数

        Returns:
            (标准照, 高清照)
        """
        height, width = target_size
        h, w = image.shape[:2]

        # 计算裁剪区域
        crop_rect = self._calculate_crop_rect(
            face_info, (w, h), target_size, layout_params
        )

        # 裁剪图像
        x1, y1, x2, y2 = crop_rect
        cropped = image[y1:y2, x1:x2]

        # 生成标准照
        standard = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)

        # 生成高清照
        hd = self._create_hd_photo(cropped, self.settings.hd_photo_min_size)

        # 水平翻转
        if layout_params.horizontal_flip:
            standard = cv2.flip(standard, 1)  # 1 表示水平翻转
            if hd is not None:
                hd = cv2.flip(hd, 1)

        return standard, hd

    def _calculate_crop_rect(
        self,
        face_info: FaceInfo,
        image_size: tuple[int, int],
        target_size: tuple[int, int],
        layout_params: LayoutParams,
    ) -> tuple[int, int, int, int]:
        """计算裁剪区域

        Args:
            face_info: 人脸信息
            image_size: 图像尺寸 (宽度, 高度)
            target_size: 目标尺寸 (高度, 宽度)
            layout_params: 布局参数

        Returns:
            裁剪区域 (x1, y1, x2, y2)
        """
        img_w, img_h = image_size
        target_h, target_w = target_size

        # 人脸中心
        face_cx, face_cy = face_info.center

        # 计算目标区域（保持宽高比）
        aspect_ratio = target_w / target_h

        # 基于人脸高度计算裁剪区域
        face_h = face_info.height
        crop_h = int(face_h / layout_params.head_measure_ratio)
        crop_w = int(crop_h * aspect_ratio)

        # 计算头顶位置
        head_top = face_info.y
        target_head_top = int(crop_h * layout_params.head_height_ratio) - (
            face_info.y - head_top
        )

        # 计算裁剪区域中心
        crop_cy = head_top + target_head_top
        crop_cx = face_cx

        # 计算裁剪边界
        x1 = max(0, crop_cx - crop_w // 2)
        y1 = max(0, crop_cy - crop_h // 2)
        x2 = min(img_w, x1 + crop_w)
        y2 = min(img_h, y1 + crop_h)

        # 调整以确保裁剪区域完整
        if x2 - x1 < crop_w:
            if x1 == 0:
                x2 = min(img_w, crop_w)
            else:
                x1 = max(0, x2 - crop_w)

        if y2 - y1 < crop_h:
            if y1 == 0:
                y2 = min(img_h, crop_h)
            else:
                y1 = max(0, y2 - crop_h)

        return (x1, y1, x2, y2)

    def _create_hd_photo(self, image: np.ndarray, min_size: int) -> np.ndarray:
        """创建高清照片

        Args:
            image: 输入图像
            min_size: 最小边长

        Returns:
            高清照片
        """
        h, w = image.shape[:2]

        # 如果图像足够大，直接返回
        if min(h, w) >= min_size:
            return image.copy()

        # 计算缩放比例
        scale = min_size / min(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # 缩放
        hd = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return hd
