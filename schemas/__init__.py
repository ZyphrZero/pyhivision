#!/usr/bin/env python

"""Data models"""

from schemas.config import (
    DetectionModelConfig,
    MattingModelConfig,
    ModelConfig,
)
from schemas.request import BeautyParams, LayoutParams, PhotoRequest
from schemas.response import FaceInfo, PhotoResult, ProcessingStats

__all__ = [
    'ModelConfig',
    'MattingModelConfig',
    'DetectionModelConfig',
    'PhotoRequest',
    'BeautyParams',
    'LayoutParams',
    'PhotoResult',
    'FaceInfo',
    'ProcessingStats',
]
