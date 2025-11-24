#!/usr/bin/env python

"""简化的模型管理器

实现线程安全的模型管理器,支持:
- 线程安全的模型加载和缓存
- LRU 淘汰策略
- 专用线程池用于 ONNX 推理
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import onnxruntime as ort

from pyhivision.config.settings import HivisionSettings
from pyhivision.exceptions.errors import ModelLoadError, ModelNotFoundError
from pyhivision.utils.logger import get_logger

logger = get_logger("core.model_manager")


class ModelManager:
    """简化的模型管理器"""

    def __init__(self, settings: HivisionSettings | None = None):
        """初始化模型管理器

        Args:
            settings: 配置实例(None 则使用默认配置)
        """
        from pyhivision.config.settings import create_settings

        self.settings = settings or create_settings()

        # 模型缓存
        self._models: dict[str, ort.InferenceSession] = {}
        self._model_lock = threading.Lock()
        self._access_order: list[str] = []  # LRU 访问顺序

        # 线程池配置
        max_workers = min(self.settings.num_threads, 8)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="onnx_inference"
        )

        logger.info(f"ModelManager initialized (thread pool: {max_workers} workers)")

    @property
    def executor(self) -> ThreadPoolExecutor:
        """获取专用线程池(供 BaseModel 使用)"""
        return self._executor

    def load_model(
        self,
        model_name: str,
        checkpoint_path: str | Path,
        use_gpu: bool | None = None,
    ) -> ort.InferenceSession:
        """加载模型(带缓存)

        Args:
            model_name: 模型名称(用作缓存键)
            checkpoint_path: 模型权重文件路径
            use_gpu: 是否使用 GPU(None 表示使用全局配置)

        Returns:
            ONNX Runtime 推理会话

        Raises:
            ModelNotFoundError: 模型文件不存在
            ModelLoadError: 模型加载失败
        """
        checkpoint_path = Path(checkpoint_path)

        with self._model_lock:
            # 检查缓存
            if model_name in self._models:
                self._update_access(model_name)
                logger.debug(f"Cache hit for model '{model_name}'")
                return self._models[model_name]

            # 缓存未命中,加载模型
            if not checkpoint_path.exists():
                raise ModelNotFoundError(model_name, str(checkpoint_path))

            self._evict_if_needed()
            return self._create_session(model_name, checkpoint_path, use_gpu)

    def _create_session(
        self,
        model_name: str,
        checkpoint_path: Path,
        use_gpu: bool | None,
    ) -> ort.InferenceSession:
        """创建 ONNX Runtime 会话"""
        import time

        start_time = time.time()

        # 确定执行提供者
        use_gpu = self.settings.enable_gpu if use_gpu is None else use_gpu
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )

        # 配置会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.log_severity_level = 3

        try:
            session = ort.InferenceSession(
                str(checkpoint_path),
                sess_options=sess_options,
                providers=providers,
            )
        except Exception as e:
            raise ModelLoadError(model_name, str(e)) from e

        load_time_ms = (time.time() - start_time) * 1000

        # 缓存会话
        self._models[model_name] = session
        self._access_order.append(model_name)

        logger.info(
            f"Model '{model_name}' loaded in {load_time_ms:.2f}ms "
            f"(providers: {session.get_providers()})"
        )

        return session

    def _update_access(self, model_name: str) -> None:
        """更新访问顺序(LRU)"""
        if model_name in self._access_order:
            self._access_order.remove(model_name)
        self._access_order.append(model_name)

    def _evict_if_needed(self) -> None:
        """LRU 淘汰策略"""
        if len(self._models) >= self.settings.model_cache_size:
            oldest_model = self._access_order.pop(0)
            logger.info(f"Evicting model '{oldest_model}' (LRU)")
            del self._models[oldest_model]

    def unload_model(self, model_name: str) -> bool:
        """卸载指定模型

        Args:
            model_name: 模型名称

        Returns:
            是否成功卸载
        """
        with self._model_lock:
            if model_name in self._models:
                del self._models[model_name]
                if model_name in self._access_order:
                    self._access_order.remove(model_name)
                logger.info(f"Model '{model_name}' unloaded")
                return True
            return False

    def clear_cache(self) -> int:
        """清空所有缓存

        Returns:
            清除的模型数量
        """
        with self._model_lock:
            count = len(self._models)
            self._models.clear()
            self._access_order.clear()
            logger.info(f"Cleared {count} models from cache")
            return count

    def shutdown(self) -> None:
        """关闭模型管理器,清理所有资源"""
        logger.info("Shutting down ModelManager...")
        self.clear_cache()
        self._executor.shutdown(wait=True)
        logger.info("ModelManager shutdown complete")
