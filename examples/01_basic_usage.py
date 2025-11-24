#!/usr/bin/env python
"""基础使用示例 - 生成标准证件照"""

import cv2
from pyhivision import IDPhotoSDK, PhotoRequest, create_settings


def main():
    # 配置模型路径
    settings = create_settings(
        matting_models_dir="~/.pyhivision/models/matting",
        detection_models_dir="~/.pyhivision/models/detection",
    )

    # 创建 SDK 实例
    sdk = IDPhotoSDK.create(settings=settings)

    # 读取图像
    image = cv2.imread("./examples/input/input_1.jpg")

    # 创建请求 - 一寸照（蓝底）
    request = PhotoRequest(
        image=image,
        size=(413, 295),  # 一寸照尺寸
        background_color=(255, 0, 0),  # 蓝色背景 (BGR)
        matting_model="modnet_photographic",
        detection_model="mtcnn",
    )

    # 处理图像
    result = sdk.process_single(request)

    # 保存结果
    cv2.imwrite("output_standard.jpg", result.standard)
    if result.hd:
        cv2.imwrite("output_hd.jpg", result.hd)

    print(f"✅ 处理完成，耗时: {result.processing_time_ms:.2f}ms")


if __name__ == "__main__":
    main()
