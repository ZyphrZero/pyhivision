#!/usr/bin/env python

"""水印处理器

支持斜向重复和居中两种水印样式。
"""
import math
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


class WatermarkStyle(str, Enum):
    """水印样式"""

    STRIPED = "striped"  # 斜向重复
    CENTRAL = "central"  # 居中


class WatermarkProcessor:
    """水印处理器"""

    def __init__(self, default_font_path: str | Path | None = None):
        """初始化水印处理器

        Args:
            default_font_path: 默认字体路径（可选）。
                如果不提供，则在调用 add_watermark 时必须提供 font_path 参数。
        """
        self.default_font_path = Path(default_font_path) if default_font_path else None

    async def add_watermark(
        self,
        image: np.ndarray,
        text: str,
        style: WatermarkStyle = WatermarkStyle.STRIPED,
        angle: int = 30,
        color: str = "#8B8B1B",
        font_path: str | Path | None = None,
        opacity: float = 0.15,
        font_size: int = 50,
        space: int = 75,
        chars_per_line: int = 8,
        font_height_crop: float = 1.2,
    ) -> np.ndarray:
        """添加水印

        Args:
            image: 输入图像 (BGR 格式)
            text: 水印文字
            style: 水印样式
            angle: 水印角度
            color: 水印颜色 (HEX 格式)
            font_path: 字体文件路径
            opacity: 透明度 [0, 1]
            font_size: 字体大小
            space: 水印间距
            chars_per_line: 每行字符数（居中模式）
            font_height_crop: 字体高度裁剪比例

        Returns:
            带水印的图像 (BGR 格式)
        """
        # 转换为 PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        pil_image = pil_image.convert("RGBA")

        # 选择字体
        if font_path is None:
            if self.default_font_path is None:
                raise ValueError(
                    "No font path provided. Either pass font_path to add_watermark() "
                    "or provide default_font_path when initializing WatermarkProcessor."
                )
            font_path = self.default_font_path

        # 应用水印
        if style == WatermarkStyle.STRIPED:
            result = self._add_watermark_striped(
                pil_image,
                text,
                angle,
                color,
                font_path,
                opacity,
                font_size,
                space,
                font_height_crop,
            )
        else:
            result = self._add_watermark_central(
                pil_image,
                text,
                angle,
                color,
                font_path,
                opacity,
                font_size,
                space,
                chars_per_line,
                font_height_crop,
            )

        # 转换回 BGR NumPy 数组
        result_rgb = result.convert("RGB")
        return cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)

    def _add_watermark_striped(
        self,
        image: Image.Image,
        text: str,
        angle: int,
        color: str,
        font_path: Path,
        opacity: float,
        font_size: int,
        space: int,
        font_height_crop: float,
    ) -> Image.Image:
        """添加斜向重复水印"""
        # 创建水印文字图像
        width = len(text) * font_size
        height = round(font_size * font_height_crop)
        watermark = Image.new("RGBA", (width, height))

        draw = ImageDraw.Draw(watermark)
        font = ImageFont.truetype(str(font_path), size=font_size)
        draw.text((0, 0), text, fill=color, font=font)

        # 裁剪边缘
        watermark = self._crop_image_edge(watermark)

        # 设置透明度
        watermark = self._set_opacity(watermark, opacity)

        # 创建水印蒙版
        c = int(math.sqrt(image.width**2 + image.height**2))
        mask = Image.new("RGBA", (c, c))

        y, idx = 0, 0
        while y < c:
            x = -int((watermark.width + space) * 0.5 * idx)
            idx = (idx + 1) % 2
            while x < c:
                mask.paste(watermark, (x, y))
                x += watermark.width + space
            y += watermark.height + space

        # 旋转蒙版
        mask = mask.rotate(angle)

        # 合成
        image.paste(
            mask,
            (int((image.width - c) / 2), int((image.height - c) / 2)),
            mask=mask.split()[3],
        )

        return image

    def _add_watermark_central(
        self,
        image: Image.Image,
        text: str,
        angle: int,
        color: str,
        font_path: Path,
        opacity: float,
        font_size: int,
        space: int,
        chars_per_line: int,
        font_height_crop: float,
    ) -> Image.Image:
        """添加居中水印"""
        import textwrap

        # 文字换行
        text_lines = textwrap.wrap(text, width=chars_per_line)
        text_wrapped = "\n".join(text_lines)

        # 创建水印文字图像
        width = len(text_wrapped) * font_size
        height = round(font_size * font_height_crop * len(text_lines))
        watermark = Image.new("RGBA", (width, height))

        draw = ImageDraw.Draw(watermark)
        font = ImageFont.truetype(str(font_path), size=font_size)
        draw.text((0, 0), text_wrapped, fill=color, font=font)

        # 裁剪边缘
        watermark = self._crop_image_edge(watermark)

        # 设置透明度
        watermark = self._set_opacity(watermark, opacity)

        # 创建水印蒙版
        c = int(math.sqrt(image.width**2 + image.height**2))
        mask = Image.new("RGBA", (c, c))
        mask.paste(
            watermark,
            (
                int((mask.width - watermark.width) / 2),
                int((mask.height - watermark.height) / 2),
            ),
        )

        # 旋转蒙版
        mask = mask.rotate(angle)

        # 合成
        image.paste(
            mask,
            (int((image.width - mask.width) / 2), int((image.height - mask.height) / 2)),
            mask=mask.split()[3],
        )

        return image

    @staticmethod
    def _set_opacity(image: Image.Image, opacity: float) -> Image.Image:
        """设置图像透明度"""
        alpha = image.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        image.putalpha(alpha)
        return image

    @staticmethod
    def _crop_image_edge(image: Image.Image) -> Image.Image:
        """裁剪图像边缘"""
        from PIL import ImageChops

        bg = Image.new("RGBA", image.size)
        diff = ImageChops.difference(image, bg)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image
