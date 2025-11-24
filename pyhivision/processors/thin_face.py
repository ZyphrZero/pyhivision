#!/usr/bin/env python

"""瘦脸处理器

使用图像局部平移法（Translation Warp）实现智能瘦脸效果。
基于人脸关键点检测，对面部左右两侧进行局部形变。

算法原理：
1. 根据人脸关键点确定变形区域（脸颊两侧）
2. 使用 cv2.remap 实现高效的局部平移
3. 强度可控，避免过度变形
"""

import math

import cv2
import numpy as np

from pyhivision.schemas.response import FaceInfo
from pyhivision.utils.logger import get_logger

logger = get_logger("processors.thin_face")


class ThinFaceProcessor:
    """瘦脸处理器

    使用局部变形算法对人脸进行瘦脸处理，需要提供人脸关键点信息。
    """

    @staticmethod
    def _local_translation_warp(
        src_image: np.ndarray,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
        radius: float,
        strength: float = 100.0,
    ) -> np.ndarray:
        """局部平移变形核心算法

        使用 cv2.remap 实现高性能的局部变形。

        Args:
            src_image: 源图像（BGR 或 BGRA 格式）
            start_point: 变形中心点 (x, y)
            end_point: 变形目标点 (x, y)
            radius: 变形半径（像素）
            strength: 变形强度，值越大变形越明显（一般 100-1000）

        Returns:
            变形后的图像

        Algorithm:
            对于变形圆内的每个像素点 (i, j)：
            1. 计算到中心点的距离 distance
            2. 计算变形比例 ratio = (R² - d²) / (R² - d² + K₀·|m-c|²)
            3. 计算映射位置 UX, UY
            4. 使用 remap 进行双线性插值

        References:
            - https://github.com/Zeyi-Lin/HivisionIDPhotos (原实现)
        """
        start_x, start_y = start_point
        end_x, end_y = end_point

        # 参数计算
        dd_radius = float(radius * radius)
        K0 = 100.0 / strength  # 强度调节因子

        # 创建圆形遮罩
        mask = np.zeros(src_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (start_x, start_y), math.ceil(radius), (255, 255, 255), -1)

        # 计算移动向量的平方
        ddmc_x = (end_x - start_x) * (end_x - start_x)
        ddmc_y = (end_y - start_y) * (end_y - start_y)

        H, W = src_image.shape[:2]

        # 生成坐标网格
        map_x = np.tile(np.arange(W, dtype=np.float32), (H, 1))
        map_y = np.tile(np.arange(H, dtype=np.float32).reshape(-1, 1), (1, W))

        # 计算每个点到中心的距离
        distance_x = (map_x - start_x) * (map_x - start_x)
        distance_y = (map_y - start_y) * (map_y - start_y)
        distance = distance_x + distance_y

        # 计算归一化距离（用于衰减）
        K1 = np.sqrt(distance)

        # 计算变形比例
        ratio_x = (dd_radius - distance_x) / (dd_radius - distance_x + K0 * ddmc_x)
        ratio_y = (dd_radius - distance_y) / (dd_radius - distance_y + K0 * ddmc_y)
        ratio_x = ratio_x * ratio_x
        ratio_y = ratio_y * ratio_y

        # 计算映射坐标（带距离衰减）
        UX = map_x - ratio_x * (end_x - start_x) * (1 - K1 / radius)
        UY = map_y - ratio_y * (end_y - start_y) * (1 - K1 / radius)

        # 在遮罩外保持原坐标
        np.copyto(UX, map_x, where=mask == 0)
        np.copyto(UY, map_y, where=mask == 0)

        # 使用 remap 进行变形
        result = cv2.remap(
            src_image, UX.astype(np.float32), UY.astype(np.float32), cv2.INTER_LINEAR
        )

        return result

    def process(
        self,
        image: np.ndarray,
        face_info: FaceInfo,
        strength: float = 3.0,
        place: int = 0,
    ) -> np.ndarray:
        """对人脸图像进行瘦脸处理

        Args:
            image: 输入图像（BGR 或 BGRA 格式）
            face_info: 人脸信息（包含 68 个关键点）
            strength: 瘦脸强度（0-10），0 表示不瘦脸，10 为最大强度
            place: 瘦脸区域选择（0-4），不同值对应不同的脸颊位置

        Returns:
            瘦脸后的图像

        Raises:
            ValueError: 如果 face_info 不包含关键点信息

        Note:
            - strength 内部会乘以 10 映射到 0-100 范围
            - 如果 strength <= 0，直接返回原图
            - 支持 3 通道（BGR）和 4 通道（BGRA）图像

        Examples:
            >>> processor = ThinFaceProcessor()
            >>> result = processor.process(image, face_info, strength=3.0)
        """
        if strength <= 0.0:
            logger.debug("Thin face strength <= 0, skipping")
            return image

        if face_info.landmarks is None or len(face_info.landmarks) < 68:
            raise ValueError("Face info must contain 68 landmarks for thin face processing")

        # 映射强度到内部范围（0-100）
        internal_strength = min(100.0, strength * 10.0)

        # 确保 place 在有效范围内
        place = max(0, min(4, int(place)))

        # 提取关键点（基于 dlib 68 点模型）
        landmarks = face_info.landmarks

        # 左脸关键点
        left_landmark = landmarks[4 + place]  # 左脸轮廓点
        left_landmark_down = landmarks[6 + place]  # 左脸下方点

        # 右脸关键点
        right_landmark = landmarks[13 + place]  # 右脸轮廓点
        right_landmark_down = landmarks[15 + place]  # 右脸下方点

        # 目标点（鼻尖位置，索引 30 或下巴中点 8）
        # 根据原代码，使用索引 58（但标准 68 点模型只到 67）
        # 这里使用鼻尖（30）作为替代
        end_point = landmarks[30] if len(landmarks) > 30 else landmarks[8]

        # 计算左侧变形半径（基于关键点距离）
        r_left = math.sqrt(
            (left_landmark[0] - left_landmark_down[0]) ** 2
            + (left_landmark[1] - left_landmark_down[1]) ** 2
        )

        # 计算右侧变形半径
        r_right = math.sqrt(
            (right_landmark[0] - right_landmark_down[0]) ** 2
            + (right_landmark[1] - right_landmark_down[1]) ** 2
        )

        logger.debug(
            f"Thin face processing: strength={internal_strength:.1f}, "
            f"place={place}, r_left={r_left:.1f}, r_right={r_right:.1f}"
        )

        # 瘦左脸
        result = self._local_translation_warp(
            image,
            start_point=(int(left_landmark[0]), int(left_landmark[1])),
            end_point=(int(end_point[0]), int(end_point[1])),
            radius=r_left,
            strength=internal_strength,
        )

        # 瘦右脸
        result = self._local_translation_warp(
            result,
            start_point=(int(right_landmark[0]), int(right_landmark[1])),
            end_point=(int(end_point[0]), int(end_point[1])),
            radius=r_right,
            strength=internal_strength,
        )

        return result
