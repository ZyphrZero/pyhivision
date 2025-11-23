#!/usr/bin/env python

"""性能监控

收集各阶段执行时间，提供统计信息。
"""
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from utils.logger import get_logger

logger = get_logger("core.metrics")


class MetricsCollector:
    """性能指标收集器"""

    def __init__(self):
        self.metrics: dict[str, list[float]] = defaultdict(list)
        self._current_request: dict[str, float] = {}

    @asynccontextmanager
    async def measure(self, stage: str):
        """测量执行时间

        Args:
            stage: 阶段名称

        Usage:
            async with metrics.measure("matting"):
                await process_matting()
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start) * 1000
            self.metrics[stage].append(elapsed_ms)
            self._current_request[stage] = elapsed_ms
            logger.debug(f"[{stage}] {elapsed_ms:.2f}ms")

    def get_current_request_times(self) -> dict[str, float]:
        """获取当前请求的各阶段耗时"""
        times = self._current_request.copy()
        self._current_request.clear()
        return times

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        stats = {}
        for stage, times in self.metrics.items():
            if not times:
                continue

            sorted_times = sorted(times)
            n = len(times)

            stats[stage] = {
                "count": n,
                "avg_ms": round(sum(times) / n, 2),
                "min_ms": round(min(times), 2),
                "max_ms": round(max(times), 2),
                "p50_ms": round(sorted_times[n // 2], 2),
                "p90_ms": round(sorted_times[int(n * 0.9)] if n >= 10 else max(times), 2),
                "p99_ms": round(sorted_times[int(n * 0.99)] if n >= 100 else max(times), 2),
            }

        return stats

    def reset(self):
        """重置统计"""
        self.metrics.clear()
        self._current_request.clear()

    def get_summary(self) -> str:
        """获取统计摘要"""
        stats = self.get_stats()
        if not stats:
            return "No metrics collected"

        lines = ["Performance Summary:"]
        for stage, data in stats.items():
            lines.append(
                f"  {stage}: avg={data['avg_ms']:.1f}ms, "
                f"p50={data['p50_ms']:.1f}ms, "
                f"p99={data['p99_ms']:.1f}ms, "
                f"count={data['count']}"
            )

        return "\n".join(lines)
