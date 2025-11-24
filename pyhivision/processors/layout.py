#!/usr/bin/env python

"""排版照片处理器

用于生成打印用的多张证件照排版图像（如 6 寸照片纸）。

"""

import cv2
import numpy as np


class LayoutProcessor:
    """排版照片处理器

    用于生成打印用的多张证件照排版图像（如 6 寸照片纸）。

    Examples:
        >>> processor = LayoutProcessor()
        >>> positions, need_rotate = processor.calculate_layout(295, 413)
        >>> photo = np.ones((413, 295, 3), dtype=np.uint8) * 255
        >>> layout = processor.generate_layout_image(photo, positions, need_rotate)
        >>> assert layout.shape == (1205, 1795, 3)
    """

    # 默认 6 寸照片纸参数（单位：像素）
    DEFAULT_LAYOUT_WIDTH = 1795
    DEFAULT_LAYOUT_HEIGHT = 1205
    DEFAULT_PHOTO_INTERVAL_H = 30  # 证件照之间的垂直间距
    DEFAULT_PHOTO_INTERVAL_W = 30  # 证件照之间的水平间距
    DEFAULT_SIDES_INTERVAL_H = 50  # 证件照与画布边缘的垂直距离
    DEFAULT_SIDES_INTERVAL_W = 70  # 证件照与画布边缘的水平距离

    def __init__(
        self,
        layout_width: int = DEFAULT_LAYOUT_WIDTH,
        layout_height: int = DEFAULT_LAYOUT_HEIGHT,
        photo_interval_h: int = DEFAULT_PHOTO_INTERVAL_H,
        photo_interval_w: int = DEFAULT_PHOTO_INTERVAL_W,
        sides_interval_h: int = DEFAULT_SIDES_INTERVAL_H,
        sides_interval_w: int = DEFAULT_SIDES_INTERVAL_W,
    ):
        """初始化排版处理器

        Args:
            layout_width: 排版画布宽度（默认 1795，6寸纸）
            layout_height: 排版画布高度（默认 1205，6寸纸）
            photo_interval_h: 照片间垂直间距
            photo_interval_w: 照片间水平间距
            sides_interval_h: 边缘垂直间距
            sides_interval_w: 边缘水平间距
        """
        self.layout_width = layout_width
        self.layout_height = layout_height
        self.photo_interval_h = photo_interval_h
        self.photo_interval_w = photo_interval_w
        self.sides_interval_h = sides_interval_h
        self.sides_interval_w = sides_interval_w

        # 计算可用区域
        self.limit_block_w = layout_width - 2 * sides_interval_w
        self.limit_block_h = layout_height - 2 * sides_interval_h

    def calculate_layout(
        self,
        photo_width: int,
        photo_height: int,
    ) -> tuple[list[tuple[int, int]], bool]:
        """计算排版位置数组

        算法：
        1. 计算不转置排列的最大照片数（列数 × 行数）
        2. 计算转置排列的最大照片数
        3. 选择照片数更多的方案

        Args:
            photo_width: 单张照片宽度（像素）
            photo_height: 单张照片高度（像素）

        Returns:
            (positions, need_rotate):
                - positions: 每张照片的 (x, y) 位置列表
                - need_rotate: 是否需要旋转照片（转置 + 垂直镜像）

        Examples:
            >>> processor = LayoutProcessor()
            >>> positions, rotate = processor.calculate_layout(295, 413)
            >>> assert len(positions) > 0
            >>> assert isinstance(rotate, bool)
        """
        # 1. 不转置排列的情况
        layout_mode_no_transpose, center_w_1, center_h_1 = self._judge_layout(
            photo_width,
            photo_height,
            transpose=False,
        )

        # 2. 转置排列的情况（宽高互换）
        layout_mode_transpose, center_w_2, center_h_2 = self._judge_layout(
            photo_height,  # 宽高互换
            photo_width,
            transpose=True,
        )

        # 3. 选择照片数更多的方案
        num_no_transpose = layout_mode_no_transpose[0] * layout_mode_no_transpose[1]
        num_transpose = layout_mode_transpose[0] * layout_mode_transpose[1]

        if num_transpose > num_no_transpose:
            layout_mode = layout_mode_transpose
            center_block_width = center_w_2
            center_block_height = center_h_2
            need_rotate = True
            # 转置模式下，宽高互换
            actual_photo_width = photo_height
            actual_photo_height = photo_width
        else:
            layout_mode = layout_mode_no_transpose
            center_block_width = center_w_1
            center_block_height = center_h_1
            need_rotate = False
            actual_photo_width = photo_width
            actual_photo_height = photo_height

        # 4. 计算起始位置（居中）
        x_start = (self.layout_width - center_block_width) // 2
        y_start = (self.layout_height - center_block_height) // 2

        # 5. 生成位置数组
        positions = []
        cols, rows = layout_mode[0], layout_mode[1]

        for row in range(rows):
            for col in range(cols):
                x = x_start + col * actual_photo_width + col * self.photo_interval_w
                y = y_start + row * actual_photo_height + row * self.photo_interval_h
                positions.append((x, y))

        return positions, need_rotate

    def _judge_layout(
        self,
        input_width: int,
        input_height: int,
        transpose: bool,
    ) -> tuple[tuple[int, int, int], int, int]:
        """判断最优排版方式

        Args:
            input_width: 照片宽度
            input_height: 照片高度
            transpose: 是否为转置模式

        Returns:
            (layout_mode, center_block_width, center_block_height):
                - layout_mode: (列数, 行数, 模式标识)
                - center_block_width: 中心区块宽度
                - center_block_height: 中心区块高度
        """
        center_block_width = input_width
        center_block_height = input_height

        # 计算行数（最多 3 行）
        layout_rows = 0
        for i in range(1, 4):
            temp_height = input_height * i + self.photo_interval_h * (i - 1)
            if temp_height < self.limit_block_h:
                center_block_height = temp_height
                layout_rows = i
            else:
                break

        # 计算列数（最多 8 列）
        layout_cols = 0
        for j in range(1, 9):
            temp_width = input_width * j + self.photo_interval_w * (j - 1)
            if temp_width < self.limit_block_w:
                center_block_width = temp_width
                layout_cols = j
            else:
                break

        layout_mode = (layout_cols, layout_rows, 2 if transpose else 1)

        return layout_mode, center_block_width, center_block_height

    def generate_layout_image(
        self,
        photo: np.ndarray,
        positions: list[tuple[int, int]],
        need_rotate: bool,
        layout_size: tuple[int, int] | None = None,
        draw_crop_lines: bool = False,
    ) -> np.ndarray:
        """生成排版图像

        Args:
            photo: 单张证件照（NumPy 数组，BGR 格式）
            positions: 位置数组（来自 calculate_layout）
            need_rotate: 是否旋转照片
            layout_size: 画布尺寸 (width, height)，默认使用初始化时的尺寸
            draw_crop_lines: 是否绘制裁剪线

        Returns:
            排版后的图像（白色背景）

        Examples:
            >>> processor = LayoutProcessor()
            >>> photo = np.ones((413, 295, 3), dtype=np.uint8) * 255
            >>> positions, rotate = processor.calculate_layout(295, 413)
            >>> layout = processor.generate_layout_image(photo, positions, rotate)
            >>> assert layout.shape[0] == 1205
        """
        if layout_size is None:
            layout_width = self.layout_width
            layout_height = self.layout_height
        else:
            layout_width, layout_height = layout_size

        # 创建白色背景画布
        white_background = np.zeros([layout_height, layout_width, 3], dtype=np.uint8)
        white_background.fill(255)

        # 获取照片尺寸
        photo_height, photo_width = photo.shape[:2]

        # 如果需要旋转，对照片进行转置和垂直镜像
        if need_rotate:
            photo = cv2.transpose(photo)
            photo = cv2.flip(photo, 0)  # 0 表示垂直镜像
            # 交换宽高
            photo_height, photo_width = photo_width, photo_height

        # 将照片按位置数组放置到画布上
        for x, y in positions:
            # 确保不越界
            if (
                y + photo_height <= layout_height
                and x + photo_width <= layout_width
            ):
                white_background[y : y + photo_height, x : x + photo_width] = photo

        # 绘制裁剪线（可选）
        if draw_crop_lines:
            self._draw_crop_lines(
                white_background,
                positions,
                photo_width,
                photo_height,
                layout_width,
                layout_height,
            )

        return white_background

    def _draw_crop_lines(
        self,
        canvas: np.ndarray,
        positions: list[tuple[int, int]],
        photo_width: int,
        photo_height: int,
        layout_width: int,
        layout_height: int,
    ) -> None:
        """绘制裁剪线

        Args:
            canvas: 画布（原地修改）
            positions: 位置数组
            photo_width: 照片宽度
            photo_height: 照片高度
            layout_width: 画布宽度
            layout_height: 画布高度
        """
        line_color = (200, 200, 200)  # 浅灰色
        line_thickness = 1

        # 收集所有裁剪线位置
        vertical_lines = set()
        horizontal_lines = set()

        for x, y in positions:
            vertical_lines.add(x)
            vertical_lines.add(x + photo_width)
            horizontal_lines.add(y)
            horizontal_lines.add(y + photo_height)

        # 绘制垂直裁剪线
        for x in vertical_lines:
            cv2.line(
                canvas,
                (x, 0),
                (x, layout_height),
                line_color,
                line_thickness,
            )

        # 绘制水平裁剪线
        for y in horizontal_lines:
            cv2.line(
                canvas,
                (0, y),
                (layout_width, y),
                line_color,
                line_thickness,
            )
