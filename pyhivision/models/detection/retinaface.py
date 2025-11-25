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

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        multiple_faces_strategy: str = "best",
    ) -> FaceInfo:
        """检测人脸（支持完整 NMS 和多人脸策略）

        Args:
            image: 输入图像 (BGR 格式)
            conf_threshold: 置信度阈值（默认 0.8）
            nms_threshold: NMS IoU 阈值（默认 0.3）
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
        from pyhivision.utils.image import nms_boxes
        from pyhivision.utils.logger import get_logger

        logger = get_logger("models.detection.retinaface")

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

        # 1. 过滤低置信度检测
        if scores is not None:
            mask = scores > conf_threshold
            boxes = boxes[mask]
            if landmarks is not None:
                landmarks = landmarks[mask]
            scores = scores[mask]
        else:
            scores = np.ones(len(boxes))

        # 检查是否有检测结果
        if len(boxes) == 0:
            raise NoFaceDetectedError()

        # 2. NMS 抑制
        if len(boxes) > 1:
            keep_indices = nms_boxes(boxes, scores, iou_threshold=nms_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            if landmarks is not None:
                landmarks = landmarks[keep_indices]

            logger.debug(f"NMS: {len(outputs[0])} → {len(boxes)} faces")

        # 3. 多人脸策略处理
        if len(boxes) > 1:
            if multiple_faces_strategy == "error":
                # 严格模式：报错
                raise MultipleFacesDetectedError(len(boxes))
            elif multiple_faces_strategy == "best":
                # 选择置信度最高的
                best_idx = np.argmax(scores)
                boxes = boxes[best_idx : best_idx + 1]
                scores = scores[best_idx : best_idx + 1]
                if landmarks is not None:
                    landmarks = landmarks[best_idx : best_idx + 1]
                logger.warning(
                    f"检测到 {len(boxes)} 个人脸，选择置信度最高的 (confidence={scores[0]:.3f})"
                )
            elif multiple_faces_strategy == "largest":
                # 选择面积最大的
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_idx = np.argmax(areas)
                boxes = boxes[largest_idx : largest_idx + 1]
                scores = scores[largest_idx : largest_idx + 1]
                if landmarks is not None:
                    landmarks = landmarks[largest_idx : largest_idx + 1]
                logger.warning(
                    f"检测到 {len(boxes)} 个人脸，选择面积最大的 (area={areas[largest_idx]:.0f})"
                )

        # 4. 提取人脸信息
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box[:4])

        # 5. 计算偏转角度
        roll_angle = 0.0
        if landmarks is not None and len(landmarks) > 0:
            landmark = landmarks[0]
            # 左眼和右眼
            left_eye = np.array([landmark[0], landmark[1]])
            right_eye = np.array([landmark[2], landmark[3]])

            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            roll_angle = float(np.degrees(np.arctan2(dy, dx)))

        # 6. 提取置信度
        confidence = float(scores[0]) if scores is not None else 1.0
        confidence = max(0.0, min(confidence, 1.0))  # 限制到 [0.0, 1.0]

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
