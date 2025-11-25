#!/usr/bin/env python

"""示例 13: 模型下载与使用

演示真实场景：首次使用时下载模型，然后处理图像。
"""
import cv2

from pyhivision import (
    IDPhotoSDK,
    PhotoRequest,
    create_settings,
    download_model,
    get_default_models_dir,
)


def main():
    """主函数"""
    print("=== PyHiVision 模型下载与使用示例 ===\n")

    # 1. 查看默认模型目录
    models_dir = get_default_models_dir()
    print(f"模型目录: {models_dir}")
    print(f"  - 抠图模型: {models_dir / 'matting'}")
    print(f"  - 检测模型: {models_dir / 'detection'}\n")

    # 2. 下载需要的模型
    print("下载模型...")
    try:
        # 下载抠图模型
        matting_path = download_model("modnet_photographic", "matting")
        print(f"✓ 抠图模型: {matting_path.name}")

        # MTCNN 使用内置权重，无需下载
        print(f"✓ 检测模型: MTCNN (内置)\n")

    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return

    # 3. 创建 SDK 并处理图像
    print("创建 SDK...")
    settings = create_settings()
    sdk = IDPhotoSDK.create(settings=settings)

    # 4. 读取并处理图像
    print("处理图像...")
    image = cv2.imread("examples/input/input_1.jpg")

    request = PhotoRequest(
        image=image,
        size=(413, 295),  # 一寸照
        background_color=(255, 0, 0),  # 蓝色背景
        matting_model="modnet_photographic",
        detection_model="mtcnn",
    )

    result = sdk.process_single(request)

    # 5. 保存结果
    cv2.imwrite("examples/output/13_standard.jpg", result.standard)
    if result.hd:
        cv2.imwrite("examples/output/13_hd.jpg", result.hd)

    print(f"✓ 处理完成！")
    print(f"  - 标准照: examples/output/13_standard.jpg")
    if result.hd:
        print(f"  - 高清照: examples/output/13_hd.jpg")


if __name__ == "__main__":
    main()
