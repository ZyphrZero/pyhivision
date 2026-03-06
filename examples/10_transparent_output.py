#!/usr/bin/env python
"""透明背景输出示例 - 生成透明背景的标准照和高清照"""

from pathlib import Path

import cv2

from pyhivision import IDPhotoSDK, PhotoRequest, create_settings


def main():
    # 配置模型路径
    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
        detection_models_dir="~/.pyhivision/detection",
    )

    # 创建 SDK 实例
    sdk = IDPhotoSDK.create(settings=settings)

    # 读取图像
    image = cv2.imread("./examples/input/input_1.jpg")

    # 创建输出目录
    output_dir = Path("examples/output/09_transparent_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 示例1：透明背景输出（BGRA 格式）
    print("=" * 60)
    print("示例1：输出透明背景的标准照和高清照")
    print("=" * 60)

    request_transparent = PhotoRequest(
        image=image,
        size=(413, 295),  # 一寸照尺寸
        matting_model="modnet_photographic",
        detection_model="mtcnn",
        add_background=False,  # 🔹关键：不添加背景，保持透明
        render_hd=True,
    )

    result_transparent = sdk.process_single(request_transparent)

    # 保存透明背景图像（PNG 格式支持 Alpha 通道）
    cv2.imwrite(str(output_dir / "standard_transparent.png"), result_transparent.standard)
    if result_transparent.hd is not None:
        cv2.imwrite(str(output_dir / "hd_transparent.png"), result_transparent.hd)

    print(f"✅ 透明背景标准照: {result_transparent.standard.shape} (BGRA)")
    print(f"✅ 透明背景高清照: {result_transparent.hd.shape} (BGRA)")
    print(f"⏱️  处理耗时: {result_transparent.processing_time_ms:.2f}ms")

    # 示例2：有背景输出（BGR 格式） - 对比
    print("\n" + "=" * 60)
    print("示例2：输出蓝色背景的标准照和高清照（对比）")
    print("=" * 60)

    request_background = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(0, 0, 255),  # 蓝色背景 (BGR)
        matting_model="modnet_photographic",
        detection_model="mtcnn",
        add_background=True,  # 🔹添加背景
        render_hd=True,
    )

    result_background = sdk.process_single(request_background)

    # 保存有背景图像
    cv2.imwrite(str(output_dir / "standard_blue.jpg"), result_background.standard)
    if result_background.hd is not None:
        cv2.imwrite(str(output_dir / "hd_blue.jpg"), result_background.hd)

    print(f"✅ 蓝色背景标准照: {result_background.standard.shape} (BGR)")
    print(f"✅ 蓝色背景高清照: {result_background.hd.shape} (BGR)")
    print(f"⏱️  处理耗时: {result_background.processing_time_ms:.2f}ms")

    # 示例3：自定义背景（后处理透明图像）
    print("\n" + "=" * 60)
    print("示例3：使用透明背景图像进行二次处理")
    print("=" * 60)

    # 读取透明背景图像
    transparent_img = result_transparent.standard

    # 方法1：添加渐变背景
    from pyhivision.processors.background import BackgroundProcessor
    bg_processor = BackgroundProcessor()

    gradient_result = bg_processor.add_gradient_background(
        transparent_img,
        start_color=(255, 200, 150),  # 浅蓝色
        end_color=(200, 150, 100),    # 浅紫色
        direction="vertical"
    )
    cv2.imwrite(str(output_dir / "standard_gradient.jpg"), gradient_result)
    print("✅ 渐变背景标准照已保存")

    # 方法2：添加自定义图像背景
    # 创建一个简单的纹理背景
    import numpy as np
    h, w = transparent_img.shape[:2]
    texture = np.zeros((h, w, 3), dtype=np.uint8)
    texture[:, :, 0] = 220  # B
    texture[:, :, 1] = 240  # G
    texture[:, :, 2] = 255  # R
    # 添加一些噪点
    noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
    texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    texture_result = bg_processor.add_image_background(transparent_img, texture)
    cv2.imwrite(str(output_dir / "standard_texture.jpg"), texture_result)
    print("✅ 纹理背景标准照已保存")

    print("\n" + "=" * 60)
    print("✨ 所有示例处理完成！")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)

    print("\n📋 输出文件列表：")
    print("  1. standard_transparent.png - 透明背景标准照 (BGRA)")
    print("  2. hd_transparent.png       - 透明背景高清照 (BGRA)")
    print("  3. standard_blue.jpg        - 蓝色背景标准照 (BGR)")
    print("  4. hd_blue.jpg              - 蓝色背景高清照 (BGR)")
    print("  5. standard_gradient.jpg    - 渐变背景标准照")
    print("  6. standard_texture.jpg     - 纹理背景标准照")


if __name__ == "__main__":
    main()
