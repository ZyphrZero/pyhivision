#!/usr/bin/env python

"""人脸矫正处理器

提供人脸角度矫正功能，基于检测到的 roll_angle 旋转图像。
支持同时处理 RGB 和 Alpha 通道，保持透明背景。
"""

import cv2
import numpy as np

from pyhivision.schemas.alignment import AlignmentParams, AlignmentResult
from pyhivision.utils.logger import get_logger

logger = get_logger("processors.alignment")


class AlignmentProcessor:
    """人脸矫正处理器

    基于人脸偏转角度（roll_angle）旋转图像，使人脸保持水平。
    旋转后画布会自动扩大以避免内容被裁剪。
    """

    def __init__(self):
        """初始化处理器"""
        logger.debug("AlignmentProcessor initialized")

    def _get_interpolation_flag(self, method: str) -> int:
        """获取 OpenCV 插值标志

        Args:
            method: 插值方法名称

        Returns:
            OpenCV 插值标志
        """
        mapping = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "nearest": cv2.INTER_NEAREST,
        }
        return mapping.get(method, cv2.INTER_LINEAR)

    def _get_border_mode_flag(self, mode: str) -> int:
        """获取 OpenCV 边界模式标志

        Args:
            mode: 边界模式名称

        Returns:
            OpenCV 边界模式标志
        """
        mapping = {
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
        }
        return mapping.get(mode, cv2.BORDER_CONSTANT)

    def _calculate_rotation_matrix(
        self,
        image_shape: tuple[int, int],
        angle: float,
        center: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, int, int, int, int]:
        """计算旋转矩阵和新画布尺寸

        基于原版 rotate_bound 函数的逻辑，计算旋转后的画布尺寸
        以确保图像内容不被裁剪。

        Args:
            image_shape: 图像尺寸 (height, width)
            angle: 旋转角度（度，正值为逆时针）
            center: 旋转中心坐标 (x, y)，None 表示使用图像中心

        Returns:
            (rotation_matrix, new_width, new_height, offset_width, offset_height)
        """
        h, w = image_shape[:2]

        # 确定旋转中心
        if center is None:
            cX, cY = w / 2.0, h / 2.0
        else:
            cX, cY = center

        # 获取旋转矩阵（注意：OpenCV 中正值为逆时针旋转）
        # 原版使用 -angle，这里为了与原版一致，也使用 -angle
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

        # 计算旋转后的边界框尺寸
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 新画布尺寸
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # 调整旋转矩阵的平移部分，使图像在新画布中居中
        M[0, 2] += (nW / 2.0) - cX
        M[1, 2] += (nH / 2.0) - cY

        # 计算偏移量
        dW = nW - w
        dH = nH - h

        return M, nW, nH, dW, dH

    def _rotate_image(
        self,
        image: np.ndarray,
        rotation_matrix: np.ndarray,
        new_size: tuple[int, int],
        params: AlignmentParams,
    ) -> np.ndarray:
        """应用旋转变换

        Args:
            image: 输入图像
            rotation_matrix: 旋转矩阵 (2x3)
            new_size: 新画布尺寸 (width, height)
            params: 矫正参数

        Returns:
            旋转后的图像
        """
        interpolation = self._get_interpolation_flag(params.interpolation)
        border_mode = self._get_border_mode_flag(params.border_mode)

        # 对于单通道图像，只使用第一个通道的边界值
        if len(image.shape) == 2:
            border_value = params.border_value[0]
        else:
            # BGR 或 BGRA 图像
            num_channels = image.shape[2]
            border_value = params.border_value[:num_channels]

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            new_size,
            flags=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )

        return rotated

    def process(
        self,
        image: np.ndarray,
        alpha: np.ndarray | None,
        roll_angle: float,
        params: AlignmentParams,
    ) -> AlignmentResult:
        """执行人脸矫正

        Args:
            image: RGB 图像 (BGR 格式，3通道)
            alpha: Alpha 通道（单通道灰度图，可选）
            roll_angle: 人脸偏转角度（度）
            params: 矫正参数

        Returns:
            矫正结果

        Raises:
            ValueError: 输入图像格式错误
        """
        # 验证输入
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel BGR image, got shape {image.shape}")

        if alpha is not None and alpha.ndim != 2:
            raise ValueError(f"Expected single-channel alpha, got shape {alpha.shape}")

        # 检查是否需要矫正
        if not params.enable_alignment:
            logger.debug("Alignment disabled by params")
            return AlignmentResult(
                rotated_image=image.copy(),
                rotated_alpha=alpha.copy() if alpha is not None else None,
                rotation_angle=0.0,
                rotation_matrix=np.eye(2, 3, dtype=np.float64),
                offset_width=0,
                offset_height=0,
                is_rotated=False,
            )

        if abs(roll_angle) <= params.angle_threshold:
            logger.debug(
                f"Roll angle {roll_angle:.2f}° is below threshold "
                f"{params.angle_threshold:.2f}°, skipping alignment"
            )
            return AlignmentResult(
                rotated_image=image.copy(),
                rotated_alpha=alpha.copy() if alpha is not None else None,
                rotation_angle=0.0,
                rotation_matrix=np.eye(2, 3, dtype=np.float64),
                offset_width=0,
                offset_height=0,
                is_rotated=False,
            )

        logger.info(f"Aligning face with roll_angle={roll_angle:.2f}°")

        # 计算旋转矩阵和新尺寸
        M, nW, nH, dW, dH = self._calculate_rotation_matrix(
            image.shape, roll_angle, params.rotation_center
        )

        # 旋转 RGB 图像
        rotated_image = self._rotate_image(image, M, (nW, nH), params)

        # 旋转 Alpha 通道（如果存在）
        rotated_alpha = None
        if alpha is not None:
            rotated_alpha = self._rotate_image(alpha, M, (nW, nH), params)

        logger.debug(
            f"Alignment complete: original size {image.shape[:2]}, "
            f"new size {rotated_image.shape[:2]}, "
            f"offset ({dW}, {dH})"
        )

        return AlignmentResult(
            rotated_image=rotated_image,
            rotated_alpha=rotated_alpha,
            rotation_angle=roll_angle,
            rotation_matrix=M,
            offset_width=dW,
            offset_height=dH,
            is_rotated=True,
        )
