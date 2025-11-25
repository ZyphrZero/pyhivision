#!/usr/bin/env python
"""自定义布局示例 - 调整人脸位置"""

from pathlib import Path

import cv2

from pyhivision import IDPhotoSDK, LayoutParams, PhotoRequest, create_settings


def main():
    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
    )
    sdk = IDPhotoSDK.create(settings=settings)

    image = cv2.imread("./examples/input/input_1.jpg")

    # 自定义布局参数
    layout = LayoutParams(
        head_measure_ratio=0.25,  # 头部占比更大
        head_height_ratio=0.5,  # 头顶位置更高
        top_distance_max=0.15,  # 增加头顶距离
    )

    request = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(255, 0, 0),
        layout_params=layout,
    )

    result = sdk.process_single(request)

    output_dir = Path("examples/output/04_layout")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "output_custom_layout.jpg"), result.standard)

    print("自定义布局完成")


if __name__ == "__main__":
    main()
