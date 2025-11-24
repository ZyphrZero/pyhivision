#!/usr/bin/env python

"""结果缓存系统

支持内存缓存和 Redis 缓存（可选）。
"""
import hashlib
import pickle
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from config.settings import HivisionSettings, create_settings
from utils.logger import get_logger

logger = get_logger("core.cache")


class CacheBackend:
    """缓存后端基类"""

    def get(self, key: str) -> Any | None:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def clear(self) -> int:
        raise NotImplementedError


class MemoryCache(CacheBackend):
    """内存缓存（LRU）"""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            self._misses += 1
            return None

        value, expire_at = self._cache[key]

        # 检查是否过期
        if expire_at and time.time() > expire_at:
            del self._cache[key]
            self._misses += 1
            return None

        # 移到末尾（LRU）
        self._cache.move_to_end(key)
        self._hits += 1

        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        # 淘汰旧条目
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        expire_at = time.time() + ttl if ttl else None
        self._cache[key] = (value, expire_at)

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "size": len(self._cache),
            "max_size": self._max_size,
        }


class ResultCache:
    """结果缓存管理器"""

    def __init__(self, settings: HivisionSettings | None = None):
        """初始化结果缓存

        Args:
            settings: 配置实例（可选）
        """
        settings = settings or create_settings()
        if settings.cache_backend == "memory":
            self._backend = MemoryCache(max_size=100)
        else:
            # TODO: 实现 Redis 后端
            self._backend = MemoryCache(max_size=100)

        self._enabled = settings.enable_cache
        self._cache_ttl = settings.cache_ttl_seconds

    def _compute_key(self, image: np.ndarray, params: dict) -> str:
        """计算缓存键

        Args:
            image: 输入图像
            params: 处理参数

        Returns:
            缓存键
        """
        # 图像哈希（采样以加速）
        h, w = image.shape[:2]
        sample = image[::max(1, h // 10), ::max(1, w // 10)]
        image_hash = hashlib.md5(sample.tobytes()).hexdigest()

        # 参数哈希
        params_str = str(sorted(params.items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()

        return f"result:{image_hash}:{params_hash}"

    def get(self, image: np.ndarray, params: dict) -> Any | None:
        """获取缓存结果"""
        if not self._enabled:
            return None

        key = self._compute_key(image, params)
        cached = self._backend.get(key)

        if cached:
            logger.debug(f"Cache hit for key: {key[:16]}...")
            return pickle.loads(cached)

        return None

    def set(self, image: np.ndarray, params: dict, result: Any) -> None:
        """设置缓存结果"""
        if not self._enabled:
            return

        key = self._compute_key(image, params)
        value = pickle.dumps(result)

        self._backend.set(key, value, ttl=self._cache_ttl)
        logger.debug(f"Cached result for key: {key[:16]}...")

    def clear(self) -> int:
        """清空缓存"""
        return self._backend.clear()

    def get_stats(self) -> dict[str, Any]:
        """获取缓存统计"""
        if hasattr(self._backend, "get_stats"):
            return self._backend.get_stats()
        return {}
