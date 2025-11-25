#!/usr/bin/env python
"""仅换背景示例 - 跳过人脸检测，仅更换背景"""

from pathlib import Path

import cv2

from pyhivision import IDPhotoSDK, PhotoRequest, create_settings


def main():
    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
    )
    sdk = IDPhotoSDK.create(settings=settings)

    image = cv2.imread("./examples/input/input_1.jpg")

    # 仅换背景模式
    request = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(0, 255, 0),  # 绿色背景
        change_bg_only=True,  # 跳过人脸检测
        render_hd=False,  # 不生成高清照
    )

    result = sdk.process_single(request)

    output_dir = Path("examples/output/08_change_bg")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "output_bg_only.jpg"), result.standard)

    print("背景更换完成")


if __name__ == "__main__":
    main()
