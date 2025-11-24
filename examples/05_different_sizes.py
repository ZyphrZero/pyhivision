#!/usr/bin/env python
"""多尺寸生成示例 - 生成不同规格证件照"""

import cv2
from pyhivision import IDPhotoSDK, PhotoRequest, create_settings


# 常见证件照尺寸（像素）
PHOTO_SIZES = {
    "一寸": (413, 295),
    "二寸": (626, 413),
    "小一寸": (390, 260),
    "护照": (354, 472),
}


def main():
    settings = create_settings(
        matting_models_dir="~/.pyhivision/models/matting",
    )
    sdk = IDPhotoSDK.create(settings=settings)

    image = cv2.imread("./examples/input/input_1.jpg")

    # 生成多种尺寸
    for name, size in PHOTO_SIZES.items():
        print(f"生成 {name} 照片 ({size[0]}x{size[1]})")

        request = PhotoRequest(
            image=image,
            size=size,
            background_color=(255, 255, 255),  # 白底
        )

        result = sdk.process_single(request)
        cv2.imwrite(f"output_{name}.jpg", result.standard)

        print(f"  ✅ 完成")


if __name__ == "__main__":
    main()
