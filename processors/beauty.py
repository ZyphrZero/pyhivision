#!/usr/bin/env python

"""美颜处理器

提供基础和高级美颜功能：
- 基础：亮度、对比度、锐化、饱和度
- 高级：美白、磨皮
"""
from pathlib import Path

import cv2
import numpy as np

from schemas.request import BeautyParams


class LutWhitening:
    """LUT 美白处理器"""

    CUBE64_ROWS = 8
    CUBE64_SIZE = 64
    CUBE256_SIZE = 256
    CUBE_SCALE = CUBE256_SIZE // CUBE64_SIZE

    def __init__(self, lut_image: np.ndarray):
        self.lut = self._create_lut(lut_image)

    def _create_lut(self, lut_image: np.ndarray) -> np.ndarray:
        """创建 3D LUT"""
        reshape_lut = np.zeros(
            (self.CUBE256_SIZE, self.CUBE256_SIZE, self.CUBE256_SIZE, 3),
            dtype=np.uint8,
        )

        for i in range(self.CUBE64_SIZE):
            tmp = i // self.CUBE64_ROWS
            cx = (i % self.CUBE64_ROWS) * self.CUBE64_SIZE
            cy = tmp * self.CUBE64_SIZE
            cube64 = lut_image[cy : cy + self.CUBE64_SIZE, cx : cx + self.CUBE64_SIZE]

            if cube64.size == 0:
                continue

            cube256 = cv2.resize(cube64, (self.CUBE256_SIZE, self.CUBE256_SIZE))
            reshape_lut[i * self.CUBE_SCALE : (i + 1) * self.CUBE_SCALE] = cube256

        return reshape_lut

    def apply(self, src: np.ndarray, strength: int) -> np.ndarray:
        """应用美白效果

        Args:
            src: 输入图像 (BGR)
            strength: 强度 [0, 30]

        Returns:
            美白后的图像
        """
        strength_normalized = np.clip(strength / 10.0, 0, 1)
        if strength_normalized <= 0:
            return src

        b, g, r = src[:, :, 0], src[:, :, 1], src[:, :, 2]
        whitened = self.lut[b, g, r]

        return cv2.addWeighted(src, 1 - strength_normalized, whitened, strength_normalized, 0)


class BeautyProcessor:
    """美颜处理器"""

    def __init__(self, lut_image_path: str | Path | None = None):
        """初始化美颜处理器

        Args:
            lut_image_path: 自定义 LUT 图像路径（可选）。
                如果不提供，使用 SDK 内置的默认 LUT。
                LUT 用于实现美白效果。
        """
        # Load LUT for whitening
        if lut_image_path is None:
            # 使用 SDK 内置的默认 LUT
            lut_path = Path(__file__).parent.parent / "assets" / "lut" / "lut_origin.png"
        else:
            # 使用自定义 LUT
            lut_path = Path(lut_image_path)

        if lut_path.exists():
            lut_image = cv2.imread(str(lut_path))
            self.whitening_processor = LutWhitening(lut_image)
        else:
            raise FileNotFoundError(f"LUT image not found: {lut_path}")

    def process(
        self,
        image: np.ndarray,
        params: BeautyParams,
    ) -> np.ndarray:
        """应用美颜效果

        Args:
            image: 输入图像 (BGR 或 BGRA 格式)
            params: 美颜参数

        Returns:
            处理后的图像
        """
        # 如果没有启用美颜，直接返回
        if not params.is_enabled:
            return image

        # 分离 alpha 通道（如果有）
        has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False
        if has_alpha:
            bgr = image[:, :, :3]
            alpha = image[:, :, 3]
        else:
            bgr = image
            alpha = None

        result = bgr.copy()

        # Advanced beauty features (apply first for better results)
        # Skin smoothing (磨皮)
        if params.skin_smoothing > 0:
            result = self._skin_smoothing(
                result,
                params.grind_degree,
                params.detail_degree,
                params.skin_smoothing,
            )

        # Whitening (美白)
        if params.whitening > 0 and self.whitening_processor:
            # Apply iteratively for stronger effect
            iteration = params.whitening // 10
            bias = params.whitening % 10

            for _ in range(iteration):
                result = self.whitening_processor.apply(result, 10)

            if bias > 0:
                result = self.whitening_processor.apply(result, bias)

        # Basic adjustments
        # 亮度调整
        if params.brightness != 0:
            result = self._adjust_brightness(result, params.brightness)

        # 对比度调整
        if params.contrast != 0:
            result = self._adjust_contrast(result, params.contrast)

        # 锐化
        if params.sharpen > 0:
            result = self._sharpen(result, params.sharpen)

        # 饱和度调整
        if params.saturation != 0:
            result = self._adjust_saturation(result, params.saturation)

        # 恢复 alpha 通道
        if has_alpha:
            result = np.dstack([result, alpha])

        return result

    def _adjust_brightness(self, image: np.ndarray, value: int) -> np.ndarray:
        """调整亮度

        Args:
            image: BGR 图像
            value: 亮度值 [-100, 100]

        Returns:
            调整后的图像
        """
        # 将值映射到合适的范围
        beta = value * 2.55  # [-255, 255]

        result = cv2.convertScaleAbs(image, alpha=1, beta=beta)
        return result

    def _adjust_contrast(self, image: np.ndarray, value: int) -> np.ndarray:
        """调整对比度

        Args:
            image: BGR 图像
            value: 对比度值 [-100, 100]

        Returns:
            调整后的图像
        """
        # 将值映射到 alpha 系数
        alpha = 1 + value / 100 if value >= 0 else 1 + value / 200

        result = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return result

    def _sharpen(self, image: np.ndarray, strength: int) -> np.ndarray:
        """锐化

        Args:
            image: BGR 图像
            strength: 锐化强度 [0, 100]

        Returns:
            锐化后的图像
        """
        # 锐化核
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)

        # 应用锐化
        sharpened = cv2.filter2D(image, -1, kernel)

        # 混合原图和锐化结果
        alpha = strength / 100
        result = cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)

        return result

    def _adjust_saturation(self, image: np.ndarray, value: int) -> np.ndarray:
        """调整饱和度

        Args:
            image: BGR 图像
            value: 饱和度值 [-100, 100]

        Returns:
            调整后的图像
        """
        # 转换到 HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 调整饱和度
        scale = 1 + value / 100 if value >= 0 else 1 + value / 200

        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)

        # 转回 BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return result

    def _skin_smoothing(
        self,
        image: np.ndarray,
        grind_degree: int,
        detail_degree: int,
        strength: int,
    ) -> np.ndarray:
        """磨皮处理

        使用双边滤波和高斯模糊实现皮肤平滑效果。

        算法：
            Dest = (Src * (100 - Opacity) +
                    (Src + 2 * GaussBlur(BilateralFilter(Src) - Src)) * Opacity) / 100

        Args:
            image: BGR 图像
            grind_degree: 磨皮程度 [1, 10]，控制双边滤波强度
            detail_degree: 细节保留程度 [1, 10]，控制高斯模糊核大小
            strength: 融合强度 [0, 10]，控制最终混合比例

        Returns:
            磨皮后的图像
        """
        if strength <= 0:
            return image

        # Normalize strength to [0, 1]
        opacity = min(10.0, strength) / 10.0

        # Calculate bilateral filter parameters
        dx = grind_degree * 5  # diameter
        fc = grind_degree * 12.5  # sigma color and space

        # Apply bilateral filter (edge-preserving smoothing)
        temp1 = cv2.bilateralFilter(image, dx, fc, fc)

        # Calculate difference
        temp2 = cv2.subtract(temp1, image)

        # Apply Gaussian blur to the difference
        kernel_size = 2 * detail_degree - 1
        temp3 = cv2.GaussianBlur(temp2, (kernel_size, kernel_size), 0)

        # Add enhanced details back
        temp4 = cv2.add(cv2.add(temp3, temp3), image)

        # Blend original and processed image
        result = cv2.addWeighted(temp4, opacity, image, 1 - opacity, 0.0)

        return result
