#!/usr/bin/env python

"""多尺寸生成示例 - 生成不同规格证件照"""

from pathlib import Path

import cv2

from pyhivision import IDPhotoSDK, PhotoRequest, create_settings

# 常见证件照尺寸（像素）
PHOTO_SIZES = {
    "1_inch": (413, 295),      # 一寸
    "2_inch": (626, 413),      # 二寸
    "small_1_inch": (390, 260),  # 小一寸
    "passport": (354, 472),    # 护照
}

# 尺寸中文名称映射（用于显示）
SIZE_NAMES_ZH = {
    "1_inch": "一寸",
    "2_inch": "二寸",
    "small_1_inch": "小一寸",
    "passport": "护照",
}


def main():
    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
    )
    sdk = IDPhotoSDK.create(settings=settings)

    image = cv2.imread("./examples/input/input_1.jpg")

    # 创建输出目录
    output_dir = Path("examples/output/05_sizes")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成多种尺寸
    for name, size in PHOTO_SIZES.items():
        name_zh = SIZE_NAMES_ZH[name]
        print(f"生成 {name_zh} 照片 ({size[0]}x{size[1]})")

        request = PhotoRequest(
            image=image,
            size=size,
            background_color=(255, 255, 255),  # 白底
        )

        result = sdk.process_single(request)
        # 使用英文文件名避免编码问题
        cv2.imwrite(str(output_dir / f"output_{name}.jpg"), result.standard)

        print(f"  完成: output_{name}.jpg")


if __name__ == "__main__":
    main()
