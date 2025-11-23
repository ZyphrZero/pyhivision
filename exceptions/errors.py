#!/usr/bin/env python

"""IDPhoto 自定义异常定义

提供清晰的异常层次结构，便于错误处理和调试。
"""
from typing import Any


class HivisionError(Exception):
    """HiVision 基础异常"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ModelLoadError(HivisionError):
    """模型加载错误"""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            details={"model_name": model_name, "reason": reason},
        )
        self.model_name = model_name


class ModelNotFoundError(ModelLoadError):
    """模型文件未找到"""

    def __init__(self, model_name: str, checkpoint_path: str):
        super().__init__(model_name, f"Checkpoint not found at '{checkpoint_path}'")
        self.checkpoint_path = checkpoint_path


class FaceDetectionError(HivisionError):
    """人脸检测错误"""

    def __init__(self, message: str, face_count: int):
        super().__init__(message, details={"face_count": face_count})
        self.face_count = face_count


class NoFaceDetectedError(FaceDetectionError):
    """未检测到人脸"""

    def __init__(self):
        super().__init__("No face detected in the image", face_count=0)


class MultipleFacesDetectedError(FaceDetectionError):
    """检测到多个人脸"""

    def __init__(self, face_count: int):
        super().__init__(
            f"Expected 1 face, but detected {face_count} faces", face_count=face_count
        )


class MattingError(HivisionError):
    """抠图错误"""

    pass


class ValidationError(HivisionError):
    """数据验证错误"""

    def __init__(self, field: str, message: str):
        super().__init__(f"Validation error on '{field}': {message}", details={"field": field})
        self.field = field


class ImageTooSmallError(ValidationError):
    """图像尺寸过小"""

    def __init__(self, width: int, height: int, min_size: int):
        super().__init__(
            "image",
            f"Image size ({width}x{height}) is smaller than minimum ({min_size}x{min_size})",
        )
        self.width = width
        self.height = height
        self.min_size = min_size


class CacheError(HivisionError):
    """缓存操作错误"""

    pass
