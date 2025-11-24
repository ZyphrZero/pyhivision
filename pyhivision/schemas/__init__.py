#!/usr/bin/env python

"""Data models"""

from pyhivision.schemas.config import (
    DetectionModelConfig,
    MattingModelConfig,
    ModelConfig,
)
from pyhivision.schemas.request import BeautyParams, LayoutParams, PhotoRequest
from pyhivision.schemas.response import FaceInfo, PhotoResult, ProcessingStats

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
