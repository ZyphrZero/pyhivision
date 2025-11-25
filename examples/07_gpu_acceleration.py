#!/usr/bin/env python
"""GPU 加速示例 - 使用 GPU 加速处理"""

from pathlib import Path

import cv2

from pyhivision import IDPhotoSDK, PhotoRequest, create_settings


def main():
    # 启用 GPU 加速
    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
        enable_gpu=True,  # 需要安装 onnxruntime-gpu
        num_threads=8,
    )

    sdk = IDPhotoSDK.create(settings=settings)

    image = cv2.imread("./examples/input/input_1.jpg")

    request = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(255, 0, 0),
    )

    result = sdk.process_single(request)

    output_dir = Path("examples/output/07_gpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "output_gpu.jpg"), result.standard)

    print(f"GPU 加速处理完成，耗时: {result.processing_time_ms:.2f}ms")


if __name__ == "__main__":
    main()
