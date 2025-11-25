#!/usr/bin/env python
"""多人脸场景处理示例

展示如何使用不同的策略处理包含多个人脸的照片：
1. 严格模式（error）：检测到多人脸时报错
2. 最佳置信度模式（best）：选择置信度最高的人脸
3. 最大面积模式（largest）：选择面积最大的人脸
"""

import sys
from pathlib import Path

import cv2

from pyhivision import IDPhotoSDK, PhotoRequest, create_settings
from pyhivision.exceptions.errors import MultipleFacesDetectedError

# 添加项目根目录到路径（开发环境）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_strict_mode(sdk, image):
    """测试严格模式：检测到多人脸时报错"""
    print("\n1. 测试严格模式（multiple_faces_strategy='error'）")
    print("=" * 60)

    try:
        request = PhotoRequest(
            image=image,
            size=(413, 295),
            background_color=(255, 255, 255),
            detection_model="mtcnn",
            multiple_faces_strategy="error",  # 严格模式：检测到多人脸时报错
        )
        sdk.process_single(request)
        print("✓ 处理成功（检测到单个人脸）")
    except MultipleFacesDetectedError as e:
        print(f"✗ 检测到多人脸！{e}")
        print("  → 严格模式下，多人脸照片将被拒绝")


def test_best_confidence_mode(sdk, image):
    """测试最佳置信度模式：选择置信度最高的人脸"""
    print("\n2. 测试最佳置信度模式（multiple_faces_strategy='best'）")
    print("=" * 60)

    request = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(255, 0, 0),  # 蓝色背景
        detection_model="mtcnn",
        multiple_faces_strategy="best",  # 选择置信度最高的人脸（默认）
    )
    result = sdk.process_single(request)

    print("✓ 处理成功！选择了置信度最高的人脸")
    print(f"  → 人脸位置: ({result.face_info.x}, {result.face_info.y})")
    print(f"  → 人脸尺寸: {result.face_info.width}x{result.face_info.height}")
    print(f"  → 置信度: {result.face_info.confidence:.3f}")

    # 保存结果
    output_path = project_root / "output" / "multiple_faces_best.jpg"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), result.standard)
    print(f"  → 结果已保存: {output_path}")


def test_largest_area_mode(sdk, image):
    """测试最大面积模式：选择面积最大的人脸"""
    print("\n3. 测试最大面积模式（multiple_faces_strategy='largest'）")
    print("=" * 60)

    request = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(0, 0, 255),  # 红色背景
        detection_model="mtcnn",
        multiple_faces_strategy="largest",  # 选择面积最大的人脸
    )
    result = sdk.process_single(request)

    print("✓ 处理成功！选择了面积最大的人脸")
    print(f"  → 人脸位置: ({result.face_info.x}, {result.face_info.y})")
    print(f"  → 人脸尺寸: {result.face_info.width}x{result.face_info.height}")
    print(f"  → 人脸面积: {result.face_info.width * result.face_info.height}")

    # 保存结果
    output_path = project_root / "output" / "multiple_faces_largest.jpg"
    cv2.imwrite(str(output_path), result.standard)
    print(f"  → 结果已保存: {output_path}")


def test_nms_configuration(sdk, image):
    """测试 NMS 配置参数"""
    print("\n4. 测试 NMS 配置参数")
    print("=" * 60)

    # 使用 RetinaFace + 完整 NMS 配置
    request = PhotoRequest(
        image=image,
        size=(413, 295),
        background_color=(255, 255, 255),
        detection_model="retinaface",  # RetinaFace 支持完整的 NMS 配置
        detection_confidence_threshold=0.7,  # 降低置信度阈值（检测更多人脸）
        detection_nms_threshold=0.3,  # NMS IoU 阈值
        multiple_faces_strategy="best",
    )

    try:
        result = sdk.process_single(request)
        print("✓ RetinaFace 处理成功！")
        print("  → 置信度阈值: 0.7")
        print("  → NMS IoU 阈值: 0.3")
        print(f"  → 检测到的人脸置信度: {result.face_info.confidence:.3f}")

        # 保存结果
        output_path = project_root / "output" / "multiple_faces_retinaface.jpg"
        cv2.imwrite(str(output_path), result.standard)
        print(f"  → 结果已保存: {output_path}")
    except Exception as e:
        print(f"✗ RetinaFace 失败: {e}")
        print("  注意：RetinaFace 需要模型文件，请确保已配置 detection_models_dir")


def main():
    """主函数"""
    print("多人脸场景处理示例")
    print("=" * 60)

    # 创建配置（请修改为实际的模型路径）
    settings = create_settings(
        matting_models_dir="~/.pyhivision/matting",
        detection_models_dir="~/.pyhivision/detection",
        enable_gpu=False,
    )

    # 创建 SDK 实例
    sdk = IDPhotoSDK.create(settings=settings)

    # 读取图像（请替换为实际的多人照片路径）
    # 注意：这个示例需要一张包含多个人脸的照片
    image_path = project_root / "examples" / "input.jpg"

    if not image_path.exists():
        print(f"\n错误：找不到输入图像 {image_path}")
        print("请准备一张包含多个人脸的照片，并将其命名为 'input.jpg'")
        print("放置在 examples 目录下。")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    print(f"✓ 已加载图像: {image_path}")
    print(f"  → 尺寸: {image.shape[1]}x{image.shape[0]}")

    # 测试不同策略
    test_strict_mode(sdk, image)
    test_best_confidence_mode(sdk, image)
    test_largest_area_mode(sdk, image)
    test_nms_configuration(sdk, image)

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("\n使用说明：")
    print("1. 严格模式（error）：适用于证件照等需要单人照片的场景")
    print("2. 最佳置信度模式（best）：适用于自动选择清晰度最高的人脸（默认）")
    print("3. 最大面积模式（largest）：适用于选择照片中的主要人物")
    print("\nNMS 配置说明：")
    print("- detection_confidence_threshold: 控制检测灵敏度（0.0-1.0）")
    print("- detection_nms_threshold: 控制重叠检测框的过滤（0.0-1.0）")
    print("- 较低的阈值 = 更多检测结果（可能包含误检）")
    print("- 较高的阈值 = 更严格的检测（可能漏检）")


if __name__ == "__main__":
    main()
