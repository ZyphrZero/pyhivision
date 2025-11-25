#!/usr/bin/env python

"""图像调整处理器

负责 ID Photo 的裁剪、缩放和布局调整。
"""
import math

import cv2
import numpy as np

from pyhivision.config.settings import HivisionSettings, create_settings
from pyhivision.schemas.request import LayoutParams
from pyhivision.schemas.response import FaceInfo
from pyhivision.utils.image import detect_head_distance, get_content_box


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

    def process(
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
        # Step1. 准备人脸参数
        x, y = face_info.x, face_info.y
        w, h = face_info.width, face_info.height
        height, width = image.shape[:2]
        standard_size = target_size  # (高度, 宽度)
        width_height_ratio = standard_size[0] / standard_size[1]

        # Step2. 计算高级参数
        face_center = (x + w / 2, y + h / 2)  # 面部中心坐标
        face_measure = w * h  # 面部面积
        crop_measure = face_measure / layout_params.head_measure_ratio  # 裁剪框面积
        resize_ratio = crop_measure / (standard_size[0] * standard_size[1])  # 裁剪框缩放率
        resize_ratio_single = math.sqrt(resize_ratio)  # 长和宽的缩放率
        crop_size = (
            int(standard_size[0] * resize_ratio_single),
            int(standard_size[1] * resize_ratio_single),
        )  # 裁剪框大小 (高度, 宽度)

        # 裁剪框的定位信息
        x1 = int(face_center[0] - crop_size[1] / 2)
        y1 = int(face_center[1] - crop_size[0] * layout_params.head_height_ratio)
        y2 = y1 + crop_size[0]
        x2 = x1 + crop_size[1]

        # Step3. 第一次裁剪
        cut_image = self._idphotos_cut(x1, y1, x2, y2, image)
        cut_image = cv2.resize(cut_image, (crop_size[1], crop_size[0]))

        # Step4. 检测裁剪后人像的实际位置
        y_top, y_bottom, x_left, x_right = get_content_box(
            cut_image.astype(np.uint8), model=2, correction_factor=0
        )

        # Step5. 判定 cut_image 中的人像是否处于合理的位置
        # 检测人像与裁剪框左边或右边是否存在空隙
        if x_left > 0 or x_right > 0:
            status_left_right = 1
            cut_value_top = int(((x_left + x_right) * width_height_ratio) / 2)
        else:
            status_left_right = 0
            cut_value_top = 0

        # 检测人头顶与照片的顶部是否在合适的距离内
        status_top, move_value = detect_head_distance(
            y_top - cut_value_top,
            crop_size[0],
            max_ratio=layout_params.top_distance_max,
            min_ratio=layout_params.top_distance_min,
        )

        # Step6. 对照片的第二轮裁剪
        if status_left_right == 0 and status_top == 0:
            result_image = cut_image
        else:
            result_image = self._idphotos_cut(
                x1 + x_left,
                y1 + cut_value_top + status_top * move_value,
                x2 - x_right,
                y2 - cut_value_top + status_top * move_value,
                image,
            )

        # Step7. 当照片底部存在空隙时，下拉至底部
        result_image, y_high = self._move_to_bottom(result_image.astype(np.uint8))

        # Step7.1 水平翻转
        if layout_params.horizontal_flip:
            result_image = cv2.flip(result_image, 1)

        # Step8. 标准照与高清照转换
        result_image_standard = self._standard_photo_resize(result_image, standard_size)
        result_image_hd, resize_ratio_max = self._resize_image_by_min(
            result_image, esp=max(600, standard_size[1])
        )

        return result_image_standard, result_image_hd

    def _idphotos_cut(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img: np.ndarray,
    ) -> np.ndarray:
        """滑动裁剪（支持超出边界的裁剪框）

        在图片上进行滑动裁剪，如果裁剪框超出了图像范围，则用透明区域补位。

        Args:
            x1: 裁剪框左上角横坐标
            y1: 裁剪框左上角纵坐标
            x2: 裁剪框右下角横坐标
            y2: 裁剪框右下角纵坐标
            img: 输入图像（BGRA 格式）

        Returns:
            裁剪后的图像
        """
        crop_size = (y2 - y1, x2 - x1)

        # 计算超出部分
        temp_x_1 = 0
        temp_y_1 = 0
        temp_x_2 = 0
        temp_y_2 = 0

        if y1 < 0:
            temp_y_1 = abs(y1)
            y1 = 0
        if y2 > img.shape[0]:
            temp_y_2 = y2
            y2 = img.shape[0]
            temp_y_2 = temp_y_2 - y2

        if x1 < 0:
            temp_x_1 = abs(x1)
            x1 = 0
        if x2 > img.shape[1]:
            temp_x_2 = x2
            x2 = img.shape[1]
            temp_x_2 = temp_x_2 - x2

        # 生成一张全透明背景
        background_bgr = np.full((crop_size[0], crop_size[1]), 255, dtype=np.uint8)
        background_a = np.full((crop_size[0], crop_size[1]), 0, dtype=np.uint8)
        background = cv2.merge((background_bgr, background_bgr, background_bgr, background_a))

        # 将裁剪区域粘贴到背景上
        background[
            temp_y_1 : crop_size[0] - temp_y_2, temp_x_1 : crop_size[1] - temp_x_2
        ] = img[y1:y2, x1:x2]

        return background

    def _move_to_bottom(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        """将人像下移至底部（消除底部空隙）

        Args:
            image: 输入图像（BGRA 格式）

        Returns:
            (处理后的图像, 下移的像素数)
        """
        height, width, channels = image.shape
        y_low, y_high, _, _ = get_content_box(image, model=2)

        # 创建顶部补白（透明区域）
        base = np.zeros((y_high, width, channels), dtype=np.uint8)

        # 裁掉底部空隙
        image = image[0 : height - y_high, :, :]

        # 重新拼接（人像下移）
        image = np.concatenate((base, image), axis=0)

        return image, y_high

    def _standard_photo_resize(
        self,
        input_image: np.ndarray,
        size: tuple[int, int],
    ) -> np.ndarray:
        """多次缩放生成标准照（保持质量）

        Args:
            input_image: 输入图像（高清照）
            size: 标准照尺寸 (高度, 宽度)

        Returns:
            标准照
        """
        resize_ratio = input_image.shape[0] / size[0]
        resize_item = int(round(input_image.shape[0] / size[0]))

        if resize_ratio >= 2:
            # 多次缩放以保持质量
            result_image = input_image
            for i in range(resize_item - 1):
                if i == 0:
                    result_image = cv2.resize(
                        input_image,
                        (size[1] * (resize_item - i - 1), size[0] * (resize_item - i - 1)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    result_image = cv2.resize(
                        result_image,
                        (size[1] * (resize_item - i - 1), size[0] * (resize_item - i - 1)),
                        interpolation=cv2.INTER_AREA,
                    )
        else:
            result_image = cv2.resize(
                input_image, (size[1], size[0]), interpolation=cv2.INTER_AREA
            )

        return result_image

    def _resize_image_by_min(
        self,
        input_image: np.ndarray,
        esp: int = 600,
    ) -> tuple[np.ndarray, float]:
        """按最短边缩放图像

        Args:
            input_image: 输入图像
            esp: 缩放后的最短边长

        Returns:
            (缩放后的图像, 缩放倍率)
        """
        height, width = input_image.shape[0], input_image.shape[1]
        min_border = min(height, width)

        if min_border < esp:
            if height >= width:
                new_width = esp
                new_height = height * esp // width
            else:
                new_height = esp
                new_width = width * esp // height

            return (
                cv2.resize(input_image, (new_width, new_height), interpolation=cv2.INTER_AREA),
                new_height / height,
            )
        else:
            return input_image, 1.0
