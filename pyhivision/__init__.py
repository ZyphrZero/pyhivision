#!/usr/bin/env python

"""PyHiVision - 高性能证件照处理 SDK"""

__version__ = '1.1.0'
__author__ = 'FastParse Team'

# Core components
from pyhivision.config.settings import HivisionSettings, create_settings
from pyhivision.core.pipeline import IDPhotoSDK, PhotoPipeline

# Exceptions
from pyhivision.exceptions.errors import (
    FaceDetectionError,
    HivisionError,
    MattingError,
    ModelLoadError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    ValidationError,
)

# Data models
from pyhivision.schemas.request import BeautyParams, LayoutParams, PhotoRequest
from pyhivision.schemas.response import FaceInfo, PhotoResult

__all__ = [
    '__version__',
    # Core API
    'IDPhotoSDK',
    'PhotoPipeline',
    # Configuration
    'HivisionSettings',
    'create_settings',
    # Data models
    'PhotoRequest',
    'PhotoResult',
    'FaceInfo',
    'BeautyParams',
    'LayoutParams',
    # Exceptions
    'HivisionError',
    'ModelLoadError',
    'FaceDetectionError',
    'NoFaceDetectedError',
    'MultipleFacesDetectedError',
    'MattingError',
    'ValidationError',
]
