#!/usr/bin/env python
"""批量处理示例 - 批量生成照片"""

from pathlib import Path

import cv2

from pyhivision import IDPhotoSDK, PhotoRequest, create_settings


def main():
    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
    )
    sdk = IDPhotoSDK.create(settings=settings)

    examples_dir = Path(__file__).resolve().parent
    input_dir = examples_dir / "input"
    output_dir = examples_dir / "output" / "03_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No JPG images found in {input_dir}")

    for img_path in image_paths:
        print(f"处理: {img_path.name}")

        image = cv2.imread(str(img_path))
        request = PhotoRequest(
            image=image,
            size=(413, 295),
            background_color=(255, 0, 0),  # 红色
        )

        result = sdk.process_single(request)

        output_path = output_dir / f"{img_path.stem}_processed.jpg"
        cv2.imwrite(str(output_path), result.standard)

        print(f"  完成，耗时: {result.processing_time_ms:.2f}ms")


if __name__ == "__main__":
    main()
