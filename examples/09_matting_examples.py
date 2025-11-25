#!/usr/bin/env python
"""抠图功能示例 - 展示不同抠图模型和抠图修补功能"""

from pathlib import Path

import cv2

from pyhivision import IDPhotoSDK, PhotoRequest, create_settings

# 创建输出目录
OUTPUT_DIR = Path("examples/output/09_matting")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def example_basic_matting():
    """示例 1: 基础抠图 - 使用不同的抠图模型"""
    print("\n=== 示例 1: 基础抠图 ===")

    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
    )

    image = cv2.imread("./examples/input/input_1.jpg")

    # 测试不同的抠图模型
    models = [
        ("modnet_photographic", "ModNet 通用摄影抠图"),
        ("hivision_modnet", "HiVision 优化版"),
        ("birefnet_lite", "BiRefNet 高精度"),
        ("rmbg_1.4", "RMBG 1.4"),
    ]

    with IDPhotoSDK.create(settings=settings) as sdk:
        for model_name, description in models:
            print(f"\n测试模型: {description} ({model_name})")

            request = PhotoRequest(
                image=image,
                size=(413, 295),
                background_color=(255, 255, 255),  # 白色背景
                matting_model=model_name,
                render_matting=True,  # 返回抠图结果
            )

            try:
                result = sdk.process_single(request)

                # 保存抠图结果（BGRA 格式，带透明通道）
                output_path = OUTPUT_DIR / f"output_matting_{model_name}.png"
                cv2.imwrite(str(output_path), result.matting)

                print(f"  ✓ 抠图成功，已保存到: {output_path}")
                print(f"  - 处理时间: {result.processing_time_ms:.2f}ms")

            except Exception as e:
                print(f"  ✗ 抠图失败: {e}")


def example_matting_with_fix():
    """示例 2: 使用抠图修补功能"""
    print("\n=== 示例 2: 抠图修补功能 ===")

    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
    )

    image = cv2.imread("./examples/input/input_1.jpg")

    with IDPhotoSDK.create(settings=settings) as sdk:
        # 不使用修补
        print("\n不使用修补:")
        request_no_fix = PhotoRequest(
            image=image,
            size=(413, 295),
            background_color=(255, 255, 255),
            matting_model="hivision_modnet",
            render_matting=True,
            enable_matting_fix=False,  # 不启用修补
        )

        result_no_fix = sdk.process_single(request_no_fix)
        output_path = OUTPUT_DIR / "output_matting_no_fix.png"
        cv2.imwrite(str(output_path), result_no_fix.matting)
        print(f"  ✓ 已保存到: {output_path}")
        print(f"  - 处理时间: {result_no_fix.processing_time_ms:.2f}ms")

        # 使用修补
        print("\n使用修补:")
        request_with_fix = PhotoRequest(
            image=image,
            size=(413, 295),
            background_color=(255, 255, 255),
            matting_model="hivision_modnet",
            render_matting=True,
            enable_matting_fix=True,  # 启用修补
        )

        result_with_fix = sdk.process_single(request_with_fix)
        output_path = OUTPUT_DIR / "output_matting_with_fix.png"
        cv2.imwrite(str(output_path), result_with_fix.matting)
        print(f"  ✓ 已保存到: {output_path}")
        print(f"  - 处理时间: {result_with_fix.processing_time_ms:.2f}ms")

        print("\n提示: 对比两张图片可以看到修补效果（边缘更平滑）")


def example_matting_with_different_backgrounds():
    """示例 3: 抠图后更换不同背景"""
    print("\n=== 示例 3: 抠图后更换不同背景 ===")

    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
    )

    image = cv2.imread("./examples/input/input_1.jpg")

    # 不同的背景颜色
    backgrounds = [
        ((255, 255, 255), "白色", "white"),
        ((255, 0, 0), "蓝色", "blue"),
        ((0, 0, 255), "红色", "red"),
        ((0, 255, 0), "绿色", "green"),
        ((200, 200, 200), "灰色", "gray"),
    ]

    with IDPhotoSDK.create(settings=settings) as sdk:
        for bg_color, color_name, color_code in backgrounds:
            print(f"\n生成{color_name}背景证件照...")

            request = PhotoRequest(
                image=image,
                size=(413, 295),
                background_color=bg_color,
                matting_model="modnet_photographic",
            )

            result = sdk.process_single(request)
            output_path = OUTPUT_DIR / f"output_bg_{color_code}.jpg"
            cv2.imwrite(str(output_path), result.standard)

            print(f"  ✓ 已保存到: {output_path}")


def example_save_matting_only():
    """示例 4: 仅保存抠图结果（透明背景）"""
    print("\n=== 示例 4: 保存透明背景抠图 ===")

    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
    )

    image = cv2.imread("./examples/input/input_1.jpg")

    with IDPhotoSDK.create(settings=settings) as sdk:
        request = PhotoRequest(
            image=image,
            size=(413, 295),
            background_color=(255, 255, 255),
            matting_model="modnet_photographic",
            render_matting=True,
            change_bg_only=True,  # 跳过人脸检测，仅抠图
        )

        result = sdk.process_single(request)

        # 保存为 PNG 格式以保留透明通道
        output_path = OUTPUT_DIR / "output_transparent.png"
        cv2.imwrite(str(output_path), result.matting)

        print(f"  ✓ 透明背景抠图已保存到: {output_path}")
        print(f"  - 图像尺寸: {result.matting.shape}")
        print(f"  - 通道数: {result.matting.shape[2]} (BGRA)")
        print(f"  - 处理时间: {result.processing_time_ms:.2f}ms")


def example_compare_matting_quality():
    """示例 5: 对比不同模型的抠图质量"""
    print("\n=== 示例 5: 抠图质量对比 ===")

    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
    )

    image = cv2.imread("./examples/input/input_1.jpg")

    models = [
        "modnet_photographic",
        "hivision_modnet",
        "birefnet_lite",
        "rmbg_1.4",
    ]

    print("\n对比不同模型的抠图质量和速度:")
    print("-" * 60)
    print(f"{'模型名称':<30} {'处理时间':<15} {'输出文件'}")
    print("-" * 60)

    with IDPhotoSDK.create(settings=settings) as sdk:
        for model_name in models:
            request = PhotoRequest(
                image=image,
                size=(413, 295),
                background_color=(0, 255, 0),  # 绿色背景便于观察
                matting_model=model_name,
                render_matting=True,
            )

            try:
                result = sdk.process_single(request)
                output_path = OUTPUT_DIR / f"output_compare_{model_name}.png"
                cv2.imwrite(str(output_path), result.matting)

                print(f"{model_name:<30} {result.processing_time_ms:>8.2f}ms     {output_path}")

            except Exception as e:
                print(f"{model_name:<30} {'失败':<15} {str(e)[:30]}")

    print("-" * 60)
    print("\n提示: 对比生成的图片，观察边缘细节和头发丝的抠图效果")


def main():
    """运行所有示例"""
    print("=" * 60)
    print("PyHiVision 抠图功能示例")
    print("=" * 60)

    try:
        # 示例 1: 基础抠图
        example_basic_matting()

        # 示例 2: 抠图修补
        example_matting_with_fix()

        # 示例 3: 不同背景
        example_matting_with_different_backgrounds()

        # 示例 4: 透明背景
        example_save_matting_only()

        # 示例 5: 质量对比
        example_compare_matting_quality()

        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
