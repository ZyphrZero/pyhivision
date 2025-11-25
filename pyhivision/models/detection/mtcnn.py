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

    def detect(
        self,
        image: np.ndarray,
        scale: int = 2,
        multiple_faces_strategy: str = "best",
    ) -> FaceInfo:
        """检测人脸（支持多人脸策略）

        Args:
            image: 输入图像 (BGR 格式)
            scale: 缩小比例（加速检测）
            multiple_faces_strategy: 多人脸处理策略
                - "error": 检测到多人脸时报错（严格模式）
                - "best": 选择置信度最高的人脸（默认）
                - "largest": 选择面积最大的人脸

        Returns:
            人脸信息

        Raises:
            NoFaceDetectedError: 未检测到人脸
            MultipleFacesDetectedError: 检测到多人脸且策略为 "error"
        """
        from pyhivision.utils.logger import get_logger

        logger = get_logger("models.detection.mtcnn")

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

        # 多人脸策略处理
        if len(faces) > 1:
            num_faces = len(faces)  # 记录原始数量
            if multiple_faces_strategy == "error":
                # 严格模式：报错
                raise MultipleFacesDetectedError(num_faces)
            elif multiple_faces_strategy == "best":
                # 选择置信度最高的
                confidences = [face[4] for face in faces]
                best_idx = np.argmax(confidences)
                faces = [faces[best_idx]]
                landmarks = [landmarks[best_idx]]
                logger.warning(
                    f"检测到 {num_faces} 个人脸，选择置信度最高的 (confidence={confidences[best_idx]:.3f})"
                )
            elif multiple_faces_strategy == "largest":
                # 选择面积最大的
                areas = [(face[2] - face[0]) * (face[3] - face[1]) for face in faces]
                largest_idx = np.argmax(areas)
                faces = [faces[largest_idx]]
                landmarks = [landmarks[largest_idx]]
                logger.warning(
                    f"检测到 {num_faces} 个人脸，选择面积最大的 (area={areas[largest_idx]:.0f})"
                )

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
        confidence = max(0.0, min(confidence, 1.0))  # 限制到 [0.0, 1.0]

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
