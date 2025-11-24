#!/usr/bin/env python

"""MTCNN 人脸检测模型实现"""
import cv2
import numpy as np
from mtcnnruntime import MTCNN as MTCNNRuntime

from pyhivision.exceptions.errors import MultipleFacesDetectedError, NoFaceDetectedError
from pyhivision.models.base import BaseDetectionModel
from pyhivision.schemas.response import FaceInfo


class MTCNNModel(BaseDetectionModel):
    """MTCNN 人脸检测模型"""

    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        self._detector = None

    def get_session(self):
        """获取检测器实例"""
        if self._detector is None:
            self._detector = MTCNNRuntime()
        return self._detector

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """MTCNN 不需要特殊预处理"""
        return image, {}

    def postprocess(self, output: np.ndarray, metadata: dict) -> np.ndarray:
        """MTCNN 不需要特殊后处理"""
        return output

    def detect(self, image: np.ndarray, scale: int = 2) -> FaceInfo:
        """检测人脸

        Args:
            image: 输入图像 (BGR 格式)
            scale: 缩小比例（加速检测）

        Returns:
            人脸信息

        Raises:
            NoFaceDetectedError: 未检测到人脸
            MultipleFacesDetectedError: 检测到多个人脸
        """
        detector = self.get_session()

        h, w = image.shape[:2]

        # 缩小图像加速检测
        resized = cv2.resize(image, (w // scale, h // scale)) if scale > 1 else image

        # 检测人脸
        faces, landmarks = detector.detect(resized, thresholds=[0.8, 0.8, 0.8])

        # 如果缩小后未检测到，使用原图重试
        if len(faces) == 0 and scale > 1:
            faces, landmarks = detector.detect(image, thresholds=[0.8, 0.8, 0.8])
            scale = 1  # 标记使用原图

        # 验证人脸数量
        if len(faces) == 0:
            raise NoFaceDetectedError()

        if len(faces) > 1:
            raise MultipleFacesDetectedError(len(faces))

        # 恢复坐标（如果使用了缩放）
        face = faces[0]
        if scale > 1:
            face = [coord * scale for coord in face]

        # 计算偏转角度
        landmark = landmarks[0]
        if scale > 1:
            landmark = [coord * scale for coord in landmark]

        # 左眼和右眼的坐标
        left_eye = np.array([landmark[0], landmark[5]])
        right_eye = np.array([landmark[1], landmark[6]])

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        roll_angle = float(np.degrees(np.arctan2(dy, dx)))

        # 计算置信度（MTCNN 返回的是分数）
        confidence = float(face[4]) if len(face) > 4 else 1.0

        return FaceInfo(
            x=int(face[0]),
            y=int(face[1]),
            width=int(face[2] - face[0] + 1),
            height=int(face[3] - face[1] + 1),
            roll_angle=roll_angle,
            confidence=confidence,
        )

    def infer(self, image: np.ndarray) -> FaceInfo:
        """推理接口（调用 detect）"""
        return self.detect(image)
