#!/usr/bin/env python
"""基础使用示例 - 生成标准证件照"""

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

    # 创建请求 - 一寸照（蓝底）
    request = PhotoRequest(
        image=image,
        size=(413, 295),  # 一寸照尺寸
        background_color="#438EDB",  # 蓝色背景（支持十六进制、RGB 元组等多种格式）
        matting_model="hivision_modnet",  # 使用 hivision_modnet 模型
        detection_model="mtcnn",
    )

    # 处理图像
    result = sdk.process_single(request)

    # 保存结果
    output_dir = Path("examples/output/01_basic_usage")
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "output_standard.jpg"), result.standard)
    if result.hd is not None:
        cv2.imwrite(str(output_dir / "output_hd.jpg"), result.hd)

    print(f"处理完成，耗时: {result.processing_time_ms:.2f}ms")


if __name__ == "__main__":
    main()
