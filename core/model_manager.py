#!/usr/bin/env python

"""统一模型管理器

实现线程安全的多实例模型管理器,支持:
- 线程安全的模型加载和缓存
- LRU 淘汰策略
- TTL 过期机制
- 统计信息收集
- 专用线程池用于 ONNX 推理
- 依赖注入配置支持
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import onnxruntime as ort

from config.settings import HivisionSettings
from exceptions.errors import ModelLoadError, ModelNotFoundError
from utils.logger import get_logger

logger = get_logger("core.model_manager")


@dataclass
class ModelWrapper:
    """模型包装器,包含会话和元数据"""

    session: ort.InferenceSession
    name: str
    checkpoint_path: str
    last_used: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    load_time_ms: float = 0.0

    def update_usage(self) -> None:
        """更新使用信息"""
        self.last_used = datetime.now()
        self.usage_count += 1

    def is_expired(self, ttl_minutes: int) -> bool:
        """检查是否过期"""
        return datetime.now() - self.last_used > timedelta(minutes=ttl_minutes)


class ModelManager:
    """统一模型管理器(支持多实例)"""

    def __init__(self, settings: HivisionSettings | None = None):
        """初始化模型管理器

        Args:
            settings: 配置实例(None 则使用默认配置)
        """
        from config.settings import create_settings

        self.settings = settings or create_settings()

        # 初始化实例属性
        self._models: dict[str, ModelWrapper] = {}
        self._model_lock = threading.Lock()
        self._stats = {
            "total_loads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
        }

        # 创建专用线程池
        max_workers, onnx_intra_threads = self._calculate_thread_config()
        self._onnx_intra_threads = onnx_intra_threads
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="onnx_inference"
        )

        cpu_count = os.cpu_count() or 4
        logger.info(
            f"ModelManager initialized (CPU cores: {cpu_count}, "
            f"thread pool: {max_workers} workers, "
            f"ONNX intra threads: {onnx_intra_threads})"
        )

    def _calculate_thread_config(self) -> tuple[int, int]:
        """计算线程池配置

        Returns:
            (max_workers, onnx_intra_threads)
        """
        cpu_count = os.cpu_count() or 4

        # 线程池大小: 用户配置优先,否则根据CPU核心数自动优化
        if self.settings.num_threads != 4:  # 4是默认值
            max_workers = min(self.settings.num_threads, 32)
        else:
            # ONNX Runtime 在4-8个worker时效率最优
            max_workers = min(
                {1: 1, 2: 2, 3: 3, 4: 4}.get(cpu_count, 6 if cpu_count <= 8 else 8), 32
            )

        # ONNX内部线程数: 单会话多线程收益有限,优先增加worker数
        onnx_intra_threads = 2 if cpu_count >= 12 else 1

        return max_workers, onnx_intra_threads

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
            if wrapper := self._models.get(model_name):
                if wrapper.is_expired(self.settings.model_ttl_minutes):
                    logger.info(f"Model '{model_name}' expired, reloading...")
                    self._unload_model_internal(model_name)
                else:
                    wrapper.update_usage()
                    self._stats["cache_hits"] += 1
                    logger.debug(f"Cache hit for model '{model_name}'")
                    return wrapper.session

            # 缓存未命中,加载模型
            self._stats["cache_misses"] += 1

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
        sess_options.intra_op_num_threads = self._onnx_intra_threads
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

        # 创建包装器并缓存
        wrapper = ModelWrapper(
            session=session,
            name=model_name,
            checkpoint_path=str(checkpoint_path),
            load_time_ms=load_time_ms,
        )
        wrapper.update_usage()

        self._models[model_name] = wrapper
        self._stats["total_loads"] += 1

        logger.info(
            f"Model '{model_name}' loaded in {load_time_ms:.2f}ms "
            f"(providers: {session.get_providers()})"
        )

        return session

    def _evict_if_needed(self) -> None:
        """LRU 淘汰策略"""
        if len(self._models) >= self.settings.model_cache_size:
            oldest_model = min(self._models.items(), key=lambda x: x[1].last_used)[0]
            logger.info(f"Evicting model '{oldest_model}' (LRU)")
            self._unload_model_internal(oldest_model)
            self._stats["evictions"] += 1

    def _unload_model_internal(self, model_name: str) -> None:
        """内部卸载方法(不加锁,由调用方加锁)"""
        if wrapper := self._models.pop(model_name, None):
            try:
                # 显式清理 ONNX Runtime 会话
                session = wrapper.session

                # 结束性能分析（如果启用）
                if hasattr(session, "end_profiling"):
                    session.end_profiling()

                # 清理输入/输出元数据引用
                if hasattr(session, "_inputs_meta"):
                    delattr(session, "_inputs_meta")
                if hasattr(session, "_outputs_meta"):
                    delattr(session, "_outputs_meta")

                # 删除会话对象（触发 C++ 析构函数，释放 GPU 显存）
                del wrapper.session
                del session

                logger.debug(f"Model '{model_name}' session cleaned successfully")
            except Exception as e:
                logger.warning(f"Error cleaning session for '{model_name}': {e}")

    def unload_model(self, model_name: str) -> bool:
        """卸载指定模型

        Args:
            model_name: 模型名称

        Returns:
            是否成功卸载
        """
        with self._model_lock:
            if model_name in self._models:
                self._unload_model_internal(model_name)
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
            for model_name in list(self._models.keys()):
                self._unload_model_internal(model_name)
            logger.info(f"Cleared {count} models from cache")
            return count

    def shutdown(self) -> None:
        """关闭模型管理器,清理所有资源"""
        logger.info("Shutting down ModelManager...")
        self.clear_cache()
        self._executor.shutdown(wait=True)
        logger.info("ModelManager shutdown complete")

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        with self._model_lock:
            total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
            hit_rate = self._stats["cache_hits"] / total_requests if total_requests > 0 else 0

            return {
                **self._stats,
                "cached_models": len(self._models),
                "hit_rate": round(hit_rate, 3),
                "model_names": list(self._models.keys()),
            }

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """获取指定模型的信息"""
        with self._model_lock:
            if wrapper := self._models.get(model_name):
                return {
                    "name": wrapper.name,
                    "checkpoint_path": wrapper.checkpoint_path,
                    "last_used": wrapper.last_used.isoformat(),
                    "usage_count": wrapper.usage_count,
                    "load_time_ms": wrapper.load_time_ms,
                }
            return None
