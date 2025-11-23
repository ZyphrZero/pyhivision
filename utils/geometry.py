#!/usr/bin/env python

"""几何计算工具"""

import math


def calculate_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """计算两点之间的距离"""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def calculate_angle(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """计算两点连线与水平线的夹角（度）"""
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    return math.degrees(math.atan2(dy, dx))


def calculate_center(
    x: int, y: int, width: int, height: int
) -> tuple[int, int]:
    """计算矩形中心点"""
    return (x + width // 2, y + height // 2)


def scale_bbox(
    bbox: tuple[int, int, int, int],
    scale: float,
    image_size: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """缩放边界框

    Args:
        bbox: (x1, y1, x2, y2)
        scale: 缩放比例
        image_size: 图像尺寸 (宽度, 高度)，用于裁剪

    Returns:
        缩放后的边界框
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w // 2
    cy = y1 + h // 2

    new_w = int(w * scale)
    new_h = int(h * scale)

    new_x1 = cx - new_w // 2
    new_y1 = cy - new_h // 2
    new_x2 = new_x1 + new_w
    new_y2 = new_y1 + new_h

    # 裁剪到图像边界
    if image_size:
        img_w, img_h = image_size
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img_w, new_x2)
        new_y2 = min(img_h, new_y2)

    return (new_x1, new_y1, new_x2, new_y2)


def fit_rect_in_image(
    rect: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """调整矩形以适应图像边界

    Args:
        rect: (x, y, width, height)
        image_size: (宽度, 高度)

    Returns:
        调整后的矩形
    """
    x, y, w, h = rect
    img_w, img_h = image_size

    # 确保不超出边界
    x = max(0, min(x, img_w - w))
    y = max(0, min(y, img_h - h))

    # 如果矩形太大，缩小
    if w > img_w:
        w = img_w
        x = 0
    if h > img_h:
        h = img_h
        y = 0

    return (x, y, w, h)
