#!/usr/bin/env python

"""Utility functions"""

from utils.compression import (
    compress_to_kb,
    compress_to_kb_base64,
    save_with_dpi,
)
from utils.encoding import (
    base64_to_numpy,
    bytes_to_base64,
    numpy_to_base64,
    numpy_to_bytes,
    save_numpy_image,
)
from utils.geometry import (
    calculate_angle,
    calculate_center,
    calculate_distance,
    fit_rect_in_image,
    scale_bbox,
)
from utils.image import (
    bgr_to_rgb,
    detect_head_distance,
    ensure_bgr,
    ensure_bgra,
    get_content_box,
    hollow_out_fix,
    resize_image_by_min,
    resize_image_to_max,
    rgb_to_bgr,
    rotate_image,
    rotate_image_4channels,
    rotate_image_with_info,
)

__all__ = [
    # Image processing
    'resize_image_to_max',
    'resize_image_by_min',
    'bgr_to_rgb',
    'rgb_to_bgr',
    'ensure_bgr',
    'ensure_bgra',
    'hollow_out_fix',
    'rotate_image',
    'rotate_image_with_info',
    'rotate_image_4channels',
    'get_content_box',
    'detect_head_distance',
    # Geometry
    'calculate_distance',
    'calculate_angle',
    'calculate_center',
    'scale_bbox',
    'fit_rect_in_image',
    # Encoding
    'numpy_to_base64',
    'base64_to_numpy',
    'bytes_to_base64',
    'numpy_to_bytes',
    'save_numpy_image',
    # Compression
    'compress_to_kb',
    'compress_to_kb_base64',
    'save_with_dpi',
]
