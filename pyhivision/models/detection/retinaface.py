#!/usr/bin/env python

"""RetinaFace 人脸检测模型实现"""
from typing import Any

import cv2
import numpy as np

from pyhivision.exceptions.errors import MultipleFacesDetectedError, NoFaceDetectedError
from pyhivision.models.base import BaseDetectionModel
from pyhivision.schemas.response import FaceInfo


class RetinaFaceModel(BaseDetectionModel):
    """RetinaFace 人脸检测模型"""

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """预处理"""
        orig_h, orig_w = image.shape[:2]

        # 转换为 RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 归一化
        normalized = rgb.astype(np.float32)

        # HWC → NCHW
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        metadata = {
            "orig_size": (orig_w, orig_h),
            "orig_image": image,
        }

        return batched, metadata

    def postprocess(
        self, output: np.ndarray, metadata: dict[str, Any]
    ) -> np.ndarray:
        """后处理"""
        return output

    def detect(self, image: np.ndarray) -> FaceInfo:
        """检测人脸

        Args:
            image: 输入图像 (BGR 格式)

        Returns:
            人脸信息

        Raises:
            NoFaceDetectedError: 未检测到人脸
            MultipleFacesDetectedError: 检测到多个人脸
        """
        # 获取会话
        session = self.get_session()

        # 预处理
        input_data, metadata = self.preprocess(image)

        # 在线程池中执行推理
        future = self.model_manager.executor.submit(
            lambda: session.run(None, {session.get_inputs()[0].name: input_data})
        )
        outputs = future.result()

        # 解析输出
        # RetinaFace 输出格式: [boxes, scores, landmarks]
        boxes = outputs[0]  # (N, 4)
        scores = outputs[1] if len(outputs) > 1 else None  # (N,)
        landmarks = outputs[2] if len(outputs) > 2 else None  # (N, 10)

        # 过滤低置信度检测
        conf_threshold = 0.8
        if scores is not None:
            mask = scores > conf_threshold
            boxes = boxes[mask]
            if landmarks is not None:
                landmarks = landmarks[mask]
            scores = scores[mask]

        # 验证人脸数量
        if len(boxes) == 0:
            raise NoFaceDetectedError()

        if len(boxes) > 1:
            raise MultipleFacesDetectedError(len(boxes))

        # 提取人脸信息
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box[:4])

        # 计算偏转角度
        roll_angle = 0.0
        if landmarks is not None and len(landmarks) > 0:
            landmark = landmarks[0]
            # 左眼和右眼
            left_eye = np.array([landmark[0], landmark[1]])
            right_eye = np.array([landmark[2], landmark[3]])

            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            roll_angle = float(np.degrees(np.arctan2(dy, dx)))

        confidence = float(scores[0]) if scores is not None else 1.0

        return FaceInfo(
            x=x1,
            y=y1,
            width=x2 - x1 + 1,
            height=y2 - y1 + 1,
            roll_angle=roll_angle,
            confidence=confidence,
        )

    def infer(self, image: np.ndarray) -> FaceInfo:
        """推理接口（调用 detect）"""
        return self.detect(image)
