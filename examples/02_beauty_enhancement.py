#!/usr/bin/env python
"""美颜增强示例 - 使用美颜参数"""

import cv2
from pyhivision import BeautyParams, IDPhotoSDK, PhotoRequest, create_settings


def main():
    settings = create_settings(
        matting_models_dir="~/.pyhivision/models/matting",
    )
    sdk = IDPhotoSDK.create(settings=settings)

    image = cv2.imread("./examples/input/input_1.jpg")

    # 配置美颜参数
    beauty = BeautyParams(
        brightness=10,  # 提亮
        contrast=5,  # 增强对比度
        whitening=15,  # 美白
        skin_smoothing=5,  # 磨皮
    )

    request = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(67, 142, 219),  # 蓝底
        beauty_params=beauty,
    )

    result = sdk.process_single(request)
    cv2.imwrite("output_beauty.jpg", result.standard)

    print(f"✅ 美颜处理完成")


if __name__ == "__main__":
    main()
