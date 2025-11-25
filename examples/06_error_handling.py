#!/usr/bin/env python
"""错误处理示例 - 处理各种异常情况"""

from pathlib import Path

import cv2

from pyhivision import (
    FaceDetectionError,
    IDPhotoSDK,
    MattingError,
    NoFaceDetectedError,
    PhotoRequest,
    ValidationError,
    create_settings,
)


def process_single_image(sdk, input_path: Path, output_dir: Path):
    """处理单张图片，包含完整的错误处理"""
    print(f"\n正在处理: {input_path.name}")
    print("=" * 60)

    try:
        # 读取图像
        image = cv2.imread(str(input_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像文件: {input_path}")

        # 创建请求
        request = PhotoRequest(
            image=image,
            size=(413, 295),
            background_color=(255, 0, 0),  # 蓝底
            matting_model="hivision_modnet",
            # matting_model="rmbg_1.4",
        )

        # 处理图像
        result = sdk.process_single(request)

        # 保存结果
        output_path = output_dir / f"output_{input_path.stem}.jpg"
        cv2.imwrite(str(output_path), result.standard)

        print(f"✓ 处理成功: {output_path.name}")
        return True

    except ValidationError as e:
        print(f"✗ 数据验证失败: {e}")
        return False
    except NoFaceDetectedError:
        print("✗ 未检测到人脸，请确保照片中有清晰的人脸")
        return False
    except FaceDetectionError as e:
        print(f"✗ 人脸检测失败: {e}")
        return False
    except MattingError as e:
        print(f"✗ 抠图失败: {e}")
        return False
    except FileNotFoundError as e:
        print(f"✗ 文件错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 处理失败: {type(e).__name__}: {e}")
        return False


def main():
    # 创建 SDK 实例
    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
    )
    sdk = IDPhotoSDK.create(settings=settings)

    # 输入和输出目录
    input_dir = Path("examples/input")
    output_dir = Path("examples/output/06_error")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    # 获取所有图片文件
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"错误: 在 {input_dir} 中未找到任何图片文件")
        return

    # 处理统计
    total = len(image_files)
    success_count = 0
    failed_count = 0

    print(f"找到 {total} 张图片，开始处理...")
    print("=" * 60)

    # 逐个处理图片
    for image_file in sorted(image_files):
        if process_single_image(sdk, image_file, output_dir):
            success_count += 1
        else:
            failed_count += 1

    # 打印统计结果
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"总计: {total} 张")
    print(f"成功: {success_count} 张")
    print(f"失败: {failed_count} 张")
    print(f"成功率: {success_count/total*100:.1f}%")
    print(f"\n输出目录: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
