#!/usr/bin/env python

"""模型下载工具

提供类似 Playwright 的模型自动下载功能，根据系统平台自动下载到对应目录。
"""
import hashlib
import platform
import sys
from pathlib import Path
from typing import Literal
from urllib.request import urlopen, Request

from pyhivision.utils.logger import get_logger

logger = get_logger("utils.download")

# 模型下载配置
MODEL_REGISTRY = {
    "modnet_photographic": {
        "url": "https://github.com/ZyphrZero/pyhivision/releases/download/pyhivision_models/modnet_photographic_portrait_matting.onnx",
        "filename": "modnet_photographic_portrait_matting.onnx",
        "sha256": None,
    },
    "hivision_modnet": {
        "url": "https://github.com/ZyphrZero/pyhivision/releases/download/pyhivision_models/hivision_modnet.onnx",
        "filename": "hivision_modnet.onnx",
        "sha256": None,
    },
    "birefnet_lite": {
        "url": "https://github.com/ZyphrZero/pyhivision/releases/download/pyhivision_models/birefnet-v1-lite.onnx",
        "filename": "birefnet-v1-lite.onnx",
        "sha256": None,
    },
    "rmbg_1.4": {
        "url": "https://github.com/ZyphrZero/pyhivision/releases/download/pyhivision_models/rmbg_1.4.onnx",
        "filename": "rmbg_1.4.onnx",
        "sha256": None,
    },
    "rmbg_2.0": {
        "url": "https://github.com/ZyphrZero/pyhivision/releases/download/pyhivision_models/rmbg_2.0_q4f16.onnx",
        "filename": "rmbg_2.0_q4f16.onnx",
        "sha256": None,
    },
    "retinaface": {
        "url": "https://github.com/ZyphrZero/pyhivision/releases/download/pyhivision_models/retinaface_resnet50.onnx",
        "filename": "retinaface_resnet50.onnx",
        "sha256": None,
    },
}


def get_default_models_dir() -> Path:
    """获取默认模型目录（根据系统平台）"""
    return Path.home() / ".pyhivision"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """下载文件并显示进度"""
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = Request(url, headers={"User-Agent": "pyhivision"})

    with urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    progress = downloaded / total_size * 100
                    print(f"\r下载进度: {progress:.1f}%", end="", file=sys.stderr)

        if total_size > 0:
            print(file=sys.stderr)


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """验证文件 SHA256"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_hash


def download_model(
    model_name: str,
    model_type: Literal["matting", "detection"] = "matting",
    models_dir: Path | str | None = None,
    force: bool = False,
) -> Path:
    """下载模型文件

    Args:
        model_name: 模型名称
        model_type: 模型类型（matting 或 detection）
        models_dir: 模型目录（None 则使用默认目录）
        force: 是否强制重新下载

    Returns:
        下载的模型文件路径

    Raises:
        ValueError: 模型名称不存在
        RuntimeError: 下载失败或校验失败
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}")

    config = MODEL_REGISTRY[model_name]

    # 确定目标目录
    if models_dir is None:
        base_dir = get_default_models_dir()
    else:
        base_dir = Path(models_dir) if isinstance(models_dir, str) else models_dir

    target_dir = base_dir / model_type
    target_file = target_dir / config["filename"]

    # 检查文件是否已存在
    if target_file.exists() and not force:
        logger.info(f"模型已存在: {target_file}")
        return target_file

    # 下载文件
    logger.info(f"下载模型 {model_name} 到 {target_file}")
    try:
        download_file(config["url"], target_file)
    except Exception as e:
        if target_file.exists():
            target_file.unlink()
        raise RuntimeError(f"下载失败: {e}") from e

    # 验证 SHA256（如果提供）
    if config["sha256"]:
        logger.info("验证文件完整性...")
        if not verify_sha256(target_file, config["sha256"]):
            target_file.unlink()
            raise RuntimeError("文件校验失败")

    logger.info(f"模型下载完成: {target_file}")
    return target_file


def download_all_models(
    models_dir: Path | str | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """下载所有模型

    Args:
        models_dir: 模型目录（None 则使用默认目录）
        force: 是否强制重新下载

    Returns:
        模型名称到文件路径的映射
    """
    results = {}

    for model_name in MODEL_REGISTRY:
        model_type = "detection" if model_name == "retinaface" else "matting"
        try:
            path = download_model(model_name, model_type, models_dir, force)
            results[model_name] = path
        except Exception as e:
            logger.error(f"下载 {model_name} 失败: {e}")

    return results
