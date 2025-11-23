#!/usr/bin/env python

"""AI 模型基类和协议定义

定义所有 AI 模型的基础接口，包括预处理、推理和后处理。

- 使用 ModelManager 的专用线程池，避免线程数失控
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from core.model_manager import ModelManager
from schemas.config import ModelConfig


class BaseModel(ABC):
    """AI 模型基类"""

    def __init__(self, config: ModelConfig, model_manager: ModelManager):
        """初始化模型

        Args:
            config: 模型配置
            model_manager: 模型管理器实例
        """
        self.config = config
        self.model_manager = model_manager
        self._session = None

    async def get_session(self):
        """获取模型会话（懒加载）"""
        if self._session is None:
            self._session = self.model_manager.load_model(
                self.config.name,
                self.config.checkpoint_path,
                self.config.use_gpu,
            )
        return self._session

    @abstractmethod
    async def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """预处理

        Args:
            image: 输入图像 (BGR 格式)

        Returns:
            处理后的输入数据和元数据
        """
        pass

    @abstractmethod
    async def postprocess(
        self, output: np.ndarray, metadata: dict[str, Any]
    ) -> np.ndarray:
        """后处理

        Args:
            output: 模型输出
            metadata: 预处理时保存的元数据

        Returns:
            最终结果
        """
        pass

    async def infer(self, image: np.ndarray) -> np.ndarray:
        """完整的推理流程

        Args:
            image: 输入图像 (BGR 格式)

        Returns:
            推理结果
        """
        # 预处理（CPU密集但GIL部分释放）
        input_data, metadata = await self.preprocess(image)

        # 获取会话
        session = await self.get_session()

        # ✅ 在专用线程池中执行推理（ONNX Runtime完全释放GIL）
        loop = asyncio.get_event_loop()
        executor = self.model_manager.executor

        output = await loop.run_in_executor(
            executor,
            lambda: session.run(None, {session.get_inputs()[0].name: input_data}),
        )

        # 后处理（CPU密集但GIL部分释放）
        result = await self.postprocess(output[0], metadata)

        return result


class BaseMattingModel(BaseModel):
    """抠图模型基类"""

    @abstractmethod
    async def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """预处理图像用于抠图"""
        pass

    @abstractmethod
    async def postprocess(
        self, output: np.ndarray, metadata: dict[str, Any]
    ) -> np.ndarray:
        """后处理抠图结果，返回 BGRA 图像"""
        pass


class BaseDetectionModel(BaseModel):
    """检测模型基类"""

    @abstractmethod
    async def detect(self, image: np.ndarray) -> Any:
        """检测人脸

        Args:
            image: 输入图像 (BGR 格式)

        Returns:
            检测结果（FaceInfo 或其他格式）
        """
        pass
