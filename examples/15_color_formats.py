#!/usr/bin/env python
"""颜色格式示例 - 展示多种颜色输入方式"""

from pathlib import Path

import cv2

from pyhivision import IDPhotoSDK, PhotoRequest, create_settings


def main():
    # 配置模型路径
    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
        detection_models_dir="~/.pyhivision/detection",
        auto_download_models=True
    )

    # 创建 SDK 实例
    sdk = IDPhotoSDK.create(settings=settings)

    # 读取图像
    image = cv2.imread("./examples/input/input_1.jpg")

    # 创建输出目录
    output_dir = Path("examples/output/15_color_formats")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PyHiVision 颜色格式示例")
    print("=" * 60)

    # 示例 1: RGB 元组（推荐，默认）
    print("\n1️⃣  RGB 元组格式（推荐）")
    print("   background_color=(255, 0, 0)  # 红色")
    request_rgb = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(255, 0, 0),  # 红色 (R, G, B)
        color_format="RGB",  # 默认值，可省略
        matting_model="hivision_modnet",
        detection_model="mtcnn",
    )
    result_rgb = sdk.process_single(request_rgb)
    cv2.imwrite(str(output_dir / "01_rgb_red.jpg"), result_rgb.standard)
    print(f"   ✅ 已保存: {output_dir / '01_rgb_red.jpg'}")

    # 示例 2: 十六进制字符串（最直观）
    print("\n2️⃣  十六进制字符串格式（最直观）")
    print("   background_color='#438EDB'  # 蓝色")
    request_hex = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color="#438EDB",  # 蓝色，支持 #RRGGBB 格式
        matting_model="hivision_modnet",
        detection_model="mtcnn",
    )
    result_hex = sdk.process_single(request_hex)
    cv2.imwrite(str(output_dir / "02_hex_blue.jpg"), result_hex.standard)
    print(f"   ✅ 已保存: {output_dir / '02_hex_blue.jpg'}")

    # 示例 3: 十六进制字符串（不带 # 前缀）
    print("\n3️⃣  十六进制字符串（不带 # 前缀）")
    print("   background_color='ADD8E6'  # 浅蓝色")
    request_hex_no_hash = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color="ADD8E6",  # 浅蓝色，不带 # 前缀也可以
        matting_model="hivision_modnet",
        detection_model="mtcnn",
    )
    result_hex_no_hash = sdk.process_single(request_hex_no_hash)
    cv2.imwrite(str(output_dir / "03_hex_light_blue.jpg"), result_hex_no_hash.standard)
    print(f"   ✅ 已保存: {output_dir / '03_hex_light_blue.jpg'}")

    # 示例 4: BGR 元组（OpenCV 原生格式）
    print("\n4️⃣  BGR 元组格式（OpenCV 原生）")
    print("   background_color=(0, 128, 0), color_format='BGR'  # 绿色")
    request_bgr = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(0, 128, 0),  # 绿色 (B, G, R)
        color_format="BGR",
        matting_model="hivision_modnet",
        detection_model="mtcnn",
    )
    result_bgr = sdk.process_single(request_bgr)
    cv2.imwrite(str(output_dir / "04_bgr_green.jpg"), result_bgr.standard)
    print(f"   ✅ 已保存: {output_dir / '04_bgr_green.jpg'}")

    # 示例 5: 常见证件照颜色
    print("\n5️⃣  常见证件照颜色")

    colors = [
        ("白色", "#FFFFFF", "05_white.jpg"),
        ("标准蓝", "#438EDB", "06_standard_blue.jpg"),
        ("深蓝", "#0E4C92", "07_dark_blue.jpg"),
        ("红色", "#FF0000", "08_red.jpg"),
        ("灰色", "#808080", "09_gray.jpg"),
    ]

    for color_name, hex_color, filename in colors:
        request = PhotoRequest(
            image=image,
            size=(413, 295),
            background_color=hex_color,
            matting_model="hivision_modnet",
            detection_model="mtcnn",
        )
        result = sdk.process_single(request)
        cv2.imwrite(str(output_dir / filename), result.standard)
        print(f"   {color_name:6s} ({hex_color}): {filename}")

    print("\n" + "=" * 60)
    print("✅ 所有示例已完成！")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 60)

    # 总结
    print("\n💡 颜色格式使用建议：")
    print("   • RGB 元组：适合程序化生成颜色")
    print("   • 十六进制：适合从设计稿直接复制颜色值")
    print("   • BGR 元组：适合与 OpenCV 代码集成")


if __name__ == "__main__":
    main()
