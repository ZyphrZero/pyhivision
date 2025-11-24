#!/usr/bin/env python

"""模板处理器

用于生成社交媒体 ID Photo 模板。
"""
import json
from pathlib import Path

import cv2
import numpy as np

from pyhivision.utils.image import rotate_image


class TemplateProcessor:
    """模板处理器"""

    def __init__(self, templates_dir: str | Path | None = None):
        """初始化模板处理器

        Args:
            templates_dir: 模板资源目录路径（可选）。
                如果不提供，则无法使用模板功能。
                目录中应包含 template_config.json 和对应的模板图片文件。
        """
        if templates_dir is None:
            self.assets_dir = None
            self.config_path = None
        else:
            self.assets_dir = Path(templates_dir)
            self.config_path = self.assets_dir / "template_config.json"
        self._config = None

    @property
    def config(self) -> dict:
        """加载模板配置"""
        if self.config_path is None:
            raise RuntimeError(
                "Template directory not provided. "
                "Please provide templates_dir when initializing TemplateProcessor."
            )

        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Template config not found: {self.config_path}")

            with open(self.config_path, encoding="utf-8") as f:
                self._config = json.load(f)

        return self._config

    def generate_template_photo(
        self,
        template_name: str,
        input_image: np.ndarray,
    ) -> np.ndarray:
        """生成模板照片

        Args:
            template_name: 模板名称
            input_image: 输入图像 (BGR 格式)

        Returns:
            模板照片 (BGR 格式)
        """
        # 获取模板配置
        if template_name not in self.config:
            raise ValueError(
                f"Template '{template_name}' not found. "
                f"Available: {list(self.config.keys())}"
            )

        template_cfg = self.config[template_name]

        # 解析配置
        template_width = template_cfg["width"]
        template_height = template_cfg["height"]
        anchor_points = template_cfg["anchor_points"]
        rotation = anchor_points["rotation"]
        left_top = anchor_points["left_top"]
        right_top = anchor_points["right_top"]
        left_bottom = anchor_points["left_bottom"]
        right_bottom = anchor_points["right_bottom"]

        # 计算变换区域尺寸
        if rotation < 0:
            height = right_bottom[1] - left_top[1]
            width = right_top[0] - left_bottom[0]
        else:
            height = left_top[1] - right_bottom[1]
            width = left_bottom[0] - right_top[0]

        # 读取模板图像
        template_image_path = self.assets_dir / f"{template_name}.png"
        if not template_image_path.exists():
            raise FileNotFoundError(f"Template image not found: {template_image_path}")

        template_image = cv2.imread(str(template_image_path), cv2.IMREAD_UNCHANGED)

        # 旋转输入图像
        rotated_image = rotate_image(input_image, -rotation)
        rotated_h, rotated_w = rotated_image.shape[:2]

        # 计算缩放比例
        scale_x = width / rotated_w
        scale_y = height / rotated_h
        scale = max(scale_x, scale_y)

        # 缩放图像
        resized_image = cv2.resize(rotated_image, None, fx=scale, fy=scale)
        resized_h, resized_w = resized_image.shape[:2]

        # 创建白色背景
        result = np.full((template_height, template_width, 3), 255, dtype=np.uint8)

        # 计算粘贴位置
        paste_x = left_bottom[0]
        paste_y = left_top[1]

        # 确保不超出边界
        paste_height = min(resized_h, template_height - paste_y)
        paste_width = min(resized_w, template_width - paste_x)

        # 粘贴图像
        result[paste_y : paste_y + paste_height, paste_x : paste_x + paste_width] = (
            resized_image[:paste_height, :paste_width]
        )

        # 叠加模板
        if template_image.shape[2] == 4:  # 有 alpha 通道
            # 转换为 RGBA（如果是 BGRA）
            template_rgba = cv2.cvtColor(template_image, cv2.COLOR_BGRA2RGBA)
            alpha = template_rgba[:, :, 3] / 255.0

            # Alpha 混合
            for c in range(3):
                result[:, :, c] = (
                    result[:, :, c] * (1 - alpha) + template_rgba[:, :, c] * alpha
                )

        return result

    def list_templates(self) -> list[str]:
        """列出所有可用模板"""
        return list(self.config.keys())
