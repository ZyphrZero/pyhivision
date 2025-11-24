#!/usr/bin/env python

"""Exception definitions"""
from pyhivision.exceptions.errors import (
    CacheError,
    FaceDetectionError,
    HivisionError,
    ImageTooSmallError,
    MattingError,
    ModelLoadError,
    ModelNotFoundError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    ValidationError,
)

__all__ = [
    'HivisionError',
    'ModelLoadError',
    'ModelNotFoundError',
    'FaceDetectionError',
    'NoFaceDetectedError',
    'MultipleFacesDetectedError',
    'MattingError',
    'ValidationError',
    'ImageTooSmallError',
    'CacheError',
]
