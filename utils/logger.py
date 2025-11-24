"""
日志抽象层 - 支持多种日志后端

提供统一的日志接口

用法示例：
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing image...")
"""

import logging
from typing import Any, Protocol


class LoggerProtocol(Protocol):
    """日志接口协议"""

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """调试日志"""
        ...

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """信息日志"""
        ...

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """警告日志"""
        ...

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """错误日志"""
        ...

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """异常日志（包含堆栈跟踪）"""
        ...


class StdlibLoggerAdapter:
    """Python 标准库 logging 适配器

    遵循 Python logging 最佳实践：
    1. SDK 使用 NullHandler，默认不输出日志
    2. 由用户在应用层配置日志处理器
    3. 保持 SDK 的非侵入性

    用户启用日志示例：
        import logging
        logging.basicConfig(level=logging.INFO)
        # 或针对 pyhivision：
        logging.getLogger("pyhivision").setLevel(logging.DEBUG)
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        # SDK 最佳实践：使用 NullHandler，避免 "No handlers found" 警告
        # 用户可以在应用层通过 logging.basicConfig() 或自定义 handler 启用日志
        if not self.logger.handlers:
            self.logger.addHandler(logging.NullHandler())

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(message, *args, **kwargs)


class LoguruLoggerAdapter:
    """loguru 日志适配器"""

    def __init__(self, name: str):
        try:
            from loguru import logger

            self.logger = logger.bind(name=name)
        except ImportError as err:
            raise ImportError(
                "loguru is not installed. Please install it with: pip install loguru"
            ) from err

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(message, *args, **kwargs)


# 全局标志：是否可用 loguru
_LOGURU_AVAILABLE: bool | None = None


def _check_loguru_available() -> bool:
    """检查 loguru 是否可用（仅检查一次）"""
    global _LOGURU_AVAILABLE
    if _LOGURU_AVAILABLE is None:
        try:
            import loguru  # noqa: F401

            _LOGURU_AVAILABLE = True
        except ImportError:
            _LOGURU_AVAILABLE = False
    return _LOGURU_AVAILABLE


def get_logger(name: str) -> LoggerProtocol:
    """
    获取日志实例（工厂函数）

    优先使用 loguru（如果已安装），否则回退到标准 logging

    Args:
        name: 日志名称（通常使用 __name__）

    Returns:
        LoggerProtocol: 日志实例

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    if _check_loguru_available():
        return LoguruLoggerAdapter(name)
    else:
        return StdlibLoggerAdapter(name)


# 便捷导出
__all__ = ["get_logger", "LoggerProtocol"]
