#!/usr/bin/env python

"""PyHiVision CLI 工具

提供命令行接口用于下载模型等操作。
"""
import argparse
import sys

from pyhivision.utils.download import (
    MODEL_REGISTRY,
    download_all_models,
    download_model,
    get_default_models_dir,
)
from pyhivision.utils.logger import get_logger

logger = get_logger("cli")


def cmd_install(args):
    """安装模型命令"""
    models_dir = args.models_dir or get_default_models_dir()

    if args.all:
        # 下载所有模型
        print(f"下载所有模型到: {models_dir}")
        results = download_all_models(models_dir, force=args.force)
        print(f"\n✓ 成功下载 {len(results)} 个模型")
        return 0

    if args.model:
        # 下载指定模型
        model_name = args.model
        if model_name not in MODEL_REGISTRY:
            print(f"错误: 未知模型 '{model_name}'")
            print(f"\n可用模型: {', '.join(MODEL_REGISTRY.keys())}")
            return 1

        model_type = "detection" if model_name == "retinaface" else "matting"
        print(f"下载模型 '{model_name}' 到: {models_dir}")

        try:
            path = download_model(model_name, model_type, models_dir, force=args.force)
            print(f"\n✓ 模型已下载到: {path}")
            return 0
        except Exception as e:
            print(f"\n✗ 下载失败: {e}")
            return 1

    # 默认下载常用模型
    print(f"下载常用模型到: {models_dir}")
    common_models = ["modnet_photographic", "hivision_modnet", "retinaface"]
    for model_name in common_models:
        try:
            model_type = "matting"
            print(f"\n下载 {model_name}...")
            download_model(model_name, model_type, models_dir, force=args.force)
        except Exception as e:
            print(f"  ✗ 失败: {e}")

    print(f"\n✓ 常用模型下载完成，下载其他模型请指定，或使用 --all 参数下载所有模型")
    return 0


def cmd_list(args):
    """列出可用模型"""
    print("可用模型列表:\n")

    matting_models = [k for k in MODEL_REGISTRY.keys() if k != "retinaface"]
    detection_models = [k for k in MODEL_REGISTRY.keys() if k == "retinaface"]

    print("抠图模型 (Matting):")
    for name in matting_models:
        config = MODEL_REGISTRY[name]
        print(f"  - {name:25} {config['filename']}")

    print("\n检测模型 (Detection):")
    for name in detection_models:
        config = MODEL_REGISTRY[name]
        print(f"  - {name:25} {config['filename']}")

    print(f"\nMTCNN 使用内置权重，无需下载")
    return 0


def cmd_info(args):
    """显示模型目录信息"""
    models_dir = get_default_models_dir()
    print(f"默认模型目录: {models_dir}")
    print(f"  - 抠图模型: {models_dir / 'matting'}")
    print(f"  - 检测模型: {models_dir / 'detection'}")

    # 检查已下载的模型
    matting_dir = models_dir / "matting"
    detection_dir = models_dir / "detection"

    if matting_dir.exists():
        matting_files = list(matting_dir.glob("*.onnx"))
        print(f"\n已下载的抠图模型 ({len(matting_files)}):")
        for f in matting_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name} ({size_mb:.1f} MB)")

    if detection_dir.exists():
        detection_files = list(detection_dir.glob("*.onnx"))
        print(f"\n已下载的检测模型 ({len(detection_files)}):")
        for f in detection_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name} ({size_mb:.1f} MB)")

    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        prog="pyhivision",
        description="PyHiVision CLI - 证件照处理 SDK 命令行工具",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # install 命令
    install_parser = subparsers.add_parser("install", help="下载模型")
    install_parser.add_argument(
        "model", nargs="?", help="模型名称（不指定则下载常用模型）"
    )
    install_parser.add_argument("--all", action="store_true", help="下载所有模型")
    install_parser.add_argument(
        "--force", action="store_true", help="强制重新下载"
    )
    install_parser.add_argument(
        "--models-dir", type=str, help="自定义模型目录"
    )
    install_parser.set_defaults(func=cmd_install)

    # list 命令
    list_parser = subparsers.add_parser("list", help="列出可用模型")
    list_parser.set_defaults(func=cmd_list)

    # info 命令
    info_parser = subparsers.add_parser("info", help="显示模型目录信息")
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
