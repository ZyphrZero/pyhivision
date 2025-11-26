#!/usr/bin/env python

"""PyHiVision - 证件照处理 SDK"""

try:
    from importlib.metadata import version
    __version__ = version("pyhivision")
except Exception:
    __version__ = "unknown"

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

# Utils
from pyhivision.utils.download import download_all_models, download_model, get_default_models_dir

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
    # Utils
    'download_model',
    'download_all_models',
    'get_default_models_dir',
    # Exceptions
    'HivisionError',
    'ModelLoadError',
    'FaceDetectionError',
    'NoFaceDetectedError',
    'MultipleFacesDetectedError',
    'MattingError',
    'ValidationError',
]
