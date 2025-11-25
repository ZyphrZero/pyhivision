#!/usr/bin/env python

"""示例 14: 模型下载方式对比

演示三种模型下载方式的实际使用场景。
"""
import cv2

from pyhivision import IDPhotoSDK, PhotoRequest, create_settings, download_model


def method_1_cli():
    """推荐方式：使用 CLI 命令下载"""
    print("=== 方式 1: CLI 命令（推荐） ===\n")
    print("在命令行运行：")
    print("  $ pyhivision install modnet_photographic")
    print("  $ pyhivision install hivision_modnet\n")
    print("然后在代码中直接使用：\n")
    print("  settings = create_settings()")
    print("  sdk = IDPhotoSDK.create(settings=settings)")
    print("  result = sdk.process_single(request)\n")


def method_2_manual():
    """方式 2: 代码中手动下载"""
    print("=== 方式 2: 代码中手动下载 ===\n")

    # 手动下载模型
    print("下载模型...")
    download_model("modnet_photographic", "matting")
    print("✓ 模型下载完成\n")

    # 使用模型
    print("处理图像...")
    settings = create_settings()
    sdk = IDPhotoSDK.create(settings=settings)

    image = cv2.imread("examples/input/input_1.jpg")
    request = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(255, 0, 0),
        matting_model="modnet_photographic",
    )

    result = sdk.process_single(request)
    cv2.imwrite("examples/output/14_method2.jpg", result.standard)
    print("✓ 处理完成: examples/output/14_method2.jpg\n")


def method_3_auto():
    """方式 3: 启用自动下载（开发环境）"""
    print("=== 方式 3: 自动下载（开发环境） ===\n")

    # 启用自动下载
    settings = create_settings(auto_download_models=True)
    sdk = IDPhotoSDK.create(settings=settings)

    print("处理图像（模型不存在时自动下载）...")
    image = cv2.imread("examples/input/input_1.jpg")
    request = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(255, 0, 0),
        matting_model="modnet_photographic",
    )

    result = sdk.process_single(request)
    cv2.imwrite("examples/output/14_method3.jpg", result.standard)
    print("✓ 处理完成: examples/output/14_method3.jpg\n")


def show_error_prompt():
    """演示友好的错误提示"""
    print("=== 友好的错误提示 ===\n")

    try:
        settings = create_settings(auto_download_models=False)
        sdk = IDPhotoSDK.create(settings=settings)

        image = cv2.imread("examples/input/input_1.jpg")
        request = PhotoRequest(
            image=image,
            size=(413, 295),
            matting_model="birefnet_lite",  # 假设不存在
        )

        sdk.process_single(request)

    except FileNotFoundError as e:
        print(str(e))


def main():
    """主函数"""
    print("=" * 60)
    print("PyHiVision 模型下载方式对比")
    print("=" * 60)
    print("\n选择运行方式：")
    print("  1. CLI 命令（推荐）")
    print("  2. 代码中手动下载")
    print("  3. 自动下载（开发环境）")
    print("  4. 查看错误提示")
    print("  0. 退出")

    choice = input("\n请输入选项 (0-4): ").strip()

    if choice == "1":
        method_1_cli()
    elif choice == "2":
        method_2_manual()
    elif choice == "3":
        method_3_auto()
    elif choice == "4":
        show_error_prompt()
    elif choice == "0":
        print("退出")
    else:
        print("❌ 无效选项")


if __name__ == "__main__":
    main()
