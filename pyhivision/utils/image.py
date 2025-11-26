#!/usr/bin/env python

"""图像处理工具函数

提供常用的图像处理工具。
"""
import cv2
import numpy as np


def resize_image_to_max(image: np.ndarray, max_size: int) -> np.ndarray:
    """调整图像大小，使最大边不超过指定尺寸

    Args:
        image: 输入图像
        max_size: 最大边长

    Returns:
        调整后的图像
    """
    h, w = image.shape[:2]
    max_edge = max(h, w)

    if max_edge <= max_size:
        return image

    scale = max_size / max_edge
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_image_by_min(image: np.ndarray, min_size: int) -> np.ndarray:
    """调整图像大小，使最小边不小于指定尺寸

    Args:
        image: 输入图像
        min_size: 最小边长

    Returns:
        调整后的图像
    """
    h, w = image.shape[:2]
    min_edge = min(h, w)

    if min_edge >= min_size:
        return image

    scale = min_size / min_edge
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR 转 RGB"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """RGB 转 BGR"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def parse_color_to_bgr(
    color: tuple[int, int, int] | str, color_format: str = "RGB"
) -> tuple[int, int, int]:
    """智能解析颜色并转换为 BGR 格式

    支持多种输入格式：
    - RGB 元组: (255, 0, 0) + color_format="RGB"
    - BGR 元组: (0, 0, 255) + color_format="BGR"
    - 十六进制字符串: "#FF0000" 或 "FF0000"

    Args:
        color: 颜色值（元组或十六进制字符串）
        color_format: 元组颜色格式 ("RGB" 或 "BGR")，仅当 color 为元组时有效

    Returns:
        BGR 格式的颜色元组 (B, G, R)

    Examples:
        >>> parse_color_to_bgr((255, 0, 0), "RGB")  # 红色
        (0, 0, 255)

        >>> parse_color_to_bgr((0, 0, 255), "BGR")  # 红色
        (0, 0, 255)

        >>> parse_color_to_bgr("#FF0000")  # 红色
        (0, 0, 255)

        >>> parse_color_to_bgr("FF0000")  # 红色
        (0, 0, 255)
    """
    # 处理十六进制字符串
    if isinstance(color, str):
        # 移除可能的 # 前缀
        hex_color = color.lstrip("#")

        # 验证十六进制格式
        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color format: {color}. Expected format: #RRGGBB or RRGGBB")

        try:
            # 解析 RGB 值
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            # 十六进制默认是 RGB 格式，转换为 BGR
            return (b, g, r)
        except ValueError as e:
            raise ValueError(f"Invalid hex color format: {color}. {e}") from e

    # 处理元组格式
    if isinstance(color, tuple) and len(color) == 3:
        # 验证颜色值范围
        for i, c in enumerate(color):
            if not 0 <= c <= 255:
                raise ValueError(f"Invalid color value at index {i}: {c}. Must be in range [0, 255]")

        # 根据指定格式转换
        if color_format.upper() == "RGB":
            # RGB -> BGR: (R, G, B) -> (B, G, R)
            return (color[2], color[1], color[0])
        elif color_format.upper() == "BGR":
            # 已经是 BGR 格式
            return color
        else:
            raise ValueError(f"Invalid color_format: {color_format}. Must be 'RGB' or 'BGR'")

    raise ValueError(f"Invalid color type: {type(color)}. Expected tuple or str")


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """确保图像是 BGR 格式

    Args:
        image: 输入图像

    Returns:
        BGR 图像
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[2] == 4:
        return image[:, :, :3]

    return image


def ensure_bgra(image: np.ndarray) -> np.ndarray:
    """确保图像是 BGRA 格式

    Args:
        image: 输入图像

    Returns:
        BGRA 图像
    """
    if len(image.shape) == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    return image


def hollow_out_fix(image: np.ndarray) -> np.ndarray:
    """修复抠图结果中的空洞

    Args:
        image: BGRA 图像

    Returns:
        修复后的图像
    """
    if image.shape[2] != 4:
        return image

    alpha = image[:, :, 3]

    # 形态学闭运算填充小空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)

    # 高斯模糊平滑边缘
    smoothed = cv2.GaussianBlur(closed, (3, 3), 0)

    result = image.copy()
    result[:, :, 3] = smoothed

    return result


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """旋转图像

    Args:
        image: 输入图像
        angle: 旋转角度（度）

    Returns:
        旋转后的图像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 计算旋转矩阵
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新的边界
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # 调整旋转中心
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

    # 执行旋转
    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0) if image.shape[2] == 4 else (0, 0, 0),
    )

    return rotated


def rotate_image_with_info(
    image: np.ndarray,
    angle: float,
    center: tuple[int, int] | None = None,
) -> tuple[np.ndarray, float, float, int, int]:
    """旋转图像并返回变换信息

    用于需要记录旋转变换参数的场景（如人脸矫正后的坐标变换）。

    Args:
        image: 输入图像（3 通道 BGR）
        angle: 旋转角度（度，正值为逆时针）
        center: 旋转中心（可选，默认图像中心）

    Returns:
        (rotated_image, cos, sin, delta_width, delta_height)
            - rotated_image: 旋转后的图像
            - cos: 角度的余弦值
            - sin: 角度的正弦值
            - delta_width: 宽度变化量（新宽度 - 原宽度）
            - delta_height: 高度变化量（新高度 - 原高度）

    Note:
        这些参数用于后续人脸矫正流程的坐标变换。

    Examples:
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> rotated, cos_val, sin_val, dw, dh = rotate_image_with_info(image, 15.0)
        >>> assert rotated.shape[0] > 100  # 旋转后尺寸增大
    """
    h, w = image.shape[:2]

    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center

    # 获取旋转矩阵（注意：OpenCV 使用负角度表示逆时针）
    matrix = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)

    # 提取余弦和正弦值
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])

    # 计算新的图像尺寸
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵以保持图像完整
    matrix[0, 2] += (new_w / 2) - cx
    matrix[1, 2] += (new_h / 2) - cy

    # 执行旋转
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h))

    # 计算偏移量
    delta_w = new_w - w
    delta_h = new_h - h

    return rotated, cos, sin, delta_w, delta_h


def rotate_image_4channels(
    bgr_image: np.ndarray,
    alpha_channel: np.ndarray,
    angle: float,
    center: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """旋转 4 通道图像（分离处理 BGR 和 Alpha）

    用于同时旋转 RGB 图像和其对应的 Alpha 通道，常用于证件照人脸矫正。

    Args:
        bgr_image: 3 通道 BGR 图像
        alpha_channel: 单通道 Alpha 图像（灰度图）
        angle: 旋转角度（度）
        center: 旋转中心（可选，默认图像中心）

    Returns:
        (rotated_bgr, rotated_bgra, cos, sin, delta_width, delta_height)
            - rotated_bgr: 旋转后的 BGR 图像（3 通道）
            - rotated_bgra: 旋转后的 BGRA 图像（4 通道，合并结果）
            - cos, sin, delta_width, delta_height: 同 rotate_image_with_info

    Examples:
        >>> bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> alpha = np.ones((100, 100), dtype=np.uint8) * 255
        >>> bgr_rot, bgra_rot, cos, sin, dw, dh = rotate_image_4channels(bgr, alpha, 10.0)
        >>> assert bgra_rot.shape[2] == 4
    """
    # 旋转 BGR 图像
    rotated_bgr, cos, sin, delta_w, delta_h = rotate_image_with_info(
        bgr_image, angle, center
    )

    # 旋转 Alpha 通道
    rotated_alpha, _, _, _, _ = rotate_image_with_info(alpha_channel, angle, center)

    # 合并为 BGRA
    b, g, r = cv2.split(rotated_bgr)
    rotated_bgra = cv2.merge((b, g, r, rotated_alpha))

    return rotated_bgr, rotated_bgra, cos, sin, delta_w, delta_h


def get_content_box(
    image: np.ndarray,
    model: int = 1,
    correction_factor: int | list[int] | None = None,
    threshold: int = 127,
) -> tuple[int, int, int, int]:
    """获取四通道图像中非透明区域的边界框

    通过分析 Alpha 通道找到图像中最大连续非透明区域的边界框。

    Args:
        image: BGRA 图像（4 通道）
        model: 返回模式
            - 1: 返回坐标 (y_up, y_down, x_left, x_right)
            - 2: 返回边距 (top_margin, bottom_margin, left_margin, right_margin)
        correction_factor: 修正因子（可选）
            - int: 四边统一修正值，或仅修正左右（[0, 0, value, value]）
            - list[int]: [上, 下, 左, 右] 分别修正
            示例：[0, 0, 1, 0] 表示左边向左扩展 1 像素
        threshold: Alpha 通道二值化阈值（默认 127）

    Returns:
        根据 model 参数返回:
            - model=1: (y_up, y_down, x_left, x_right) - 边界框坐标
            - model=2: (top, bottom, left, right) - 边距值

    Raises:
        TypeError: 图像不是 4 通道或 correction_factor 类型错误
        ValueError: model 参数无效

    Examples:
        >>> # 创建带透明通道的图像
        >>> image = np.zeros((200, 200, 4), dtype=np.uint8)
        >>> image[50:150, 50:150, :] = 255  # 中心区域不透明
        >>> y1, y2, x1, x2 = get_content_box(image, model=1)
        >>> assert y1 == 50 and y2 == 150
    """
    # 数据格式验证
    if not isinstance(image, np.ndarray) or image.shape[2] != 4:
        raise TypeError("输入的图像必须为四通道 np.ndarray 类型矩阵！")

    # 规范化 correction_factor
    if correction_factor is None:
        correction_factor = [0, 0, 0, 0]
    elif isinstance(correction_factor, int):
        # int 类型：默认只修正左右
        correction_factor = [0, 0, correction_factor, correction_factor]
    elif not isinstance(correction_factor, list) or len(correction_factor) != 4:
        raise TypeError("correction_factor 必须为 int 或长度为 4 的 list！")

    # 分离 Alpha 通道
    _, _, _, alpha = cv2.split(image)

    # Alpha 通道二值化
    _, binary_mask = cv2.threshold(alpha, threshold, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # 如果没有找到轮廓，返回整个图像区域
        h, w = image.shape[:2]
        if model == 1:
            return (0, h, 0, w)
        elif model == 2:
            return (0, 0, 0, 0)
        else:
            raise ValueError(f"Invalid model: {model}. Must be 1 or 2.")

    # 找到面积最大的轮廓
    contours_area = [cv2.contourArea(cnt) for cnt in contours]
    idx = contours_area.index(max(contours_area))

    # 获取边界矩形
    x, y, w, h = cv2.boundingRect(contours[idx])

    # 应用修正因子（确保不越界）
    img_height, img_width = image.shape[:2]

    y_up = max(0, y - correction_factor[0])
    y_down = min(img_height, y + h + correction_factor[1])
    x_left = max(0, x - correction_factor[2])
    x_right = min(img_width, x + w + correction_factor[3])

    # 根据 model 返回不同格式
    if model == 1:
        # 返回坐标
        return (y_up, y_down, x_left, x_right)
    elif model == 2:
        # 返回边距
        return (y_up, img_height - y_down, x_left, img_width - x_right)
    else:
        raise ValueError(f"Invalid model: {model}. Must be 1 or 2.")


def detect_head_distance(
    value: int,
    crop_height: int,
    max_ratio: float = 0.06,
    min_ratio: float = 0.04,
) -> tuple[int, int]:
    """检测头顶距离是否合适

    用于判断头部在画面中的位置是否符合证件照规范。

    Args:
        value: 头顶到画布顶部的距离（像素）
        crop_height: 裁剪后的总高度（像素）
        max_ratio: 最大距离比例（默认 0.06，即 6%）
        min_ratio: 最小距离比例（默认 0.04，即 4%）

    Returns:
        (status, move_value):
            - status: 移动状态
                - 0: 位置合适，不需要移动
                - 1: 需要向上移动（头顶距离太大）
                - -1: 需要向下移动（头顶距离太小）
            - move_value: 需要移动的像素值（绝对值）

    Examples:
        >>> # 头顶距离占 5%，在 4%-6% 范围内，合适
        >>> status, move = detect_head_distance(20, 400, max_ratio=0.06, min_ratio=0.04)
        >>> assert status == 0 and move == 0
        >>>
        >>> # 头顶距离占 8%，超过 6%，需要上移
        >>> status, move = detect_head_distance(32, 400, max_ratio=0.06, min_ratio=0.04)
        >>> assert status == 1 and move > 0
    """
    # 计算头顶距离占比
    ratio = value / crop_height

    if min_ratio <= ratio <= max_ratio:
        # 距离合适
        return (0, 0)
    elif ratio > max_ratio:
        # 头顶距离太大，需要向上移动（裁剪框向下移动）
        move_value = ratio - max_ratio
        move_pixels = int(move_value * crop_height)
        return (1, move_pixels)
    else:
        # 头顶距离太小，需要向下移动（裁剪框向上移动）
        move_value = min_ratio - ratio
        move_pixels = int(move_value * crop_height)
        return (-1, move_pixels)


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.3,
) -> list[int]:
    """非极大值抑制（NMS）算法

    用于过滤重叠的检测框，保留最优的检测结果。

    Args:
        boxes: 检测框数组 (N, 4)，格式为 [x1, y1, x2, y2]
        scores: 置信度数组 (N,)
        iou_threshold: IoU 阈值，默认 0.3（低于此值的框会被保留）

    Returns:
        保留的检测框索引列表

    Algorithm:
        1. 按置信度降序排列
        2. 选择置信度最高的框
        3. 计算该框与其他框的 IoU
        4. 移除 IoU > threshold 的框
        5. 重复 2-4 直到所有框处理完毕

    Examples:
        >>> boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]])
        >>> scores = np.array([0.9, 0.8, 0.95])
        >>> keep = nms_boxes(boxes, scores, iou_threshold=0.3)
        >>> # 返回 [2, 0]（保留置信度最高的和不重叠的）
    """
    if len(boxes) == 0:
        return []

    # 转换为浮点数
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    # 提取坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按置信度降序排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 选择置信度最高的框
        i = order[0]
        keep.append(i)

        # 计算 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # IoU = 交集 / 并集
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU <= threshold 的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
