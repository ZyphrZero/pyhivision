"""人脸矫正示例

演示如何使用 PyHiVision 的人脸矫正功能，自动修正倾斜的人脸照片。

功能说明：
- 自动检测人脸偏转角度（roll_angle）
- 当角度超过阈值时自动旋转图像
- 矫正后重新检测人脸以更新坐标
- 支持标准模式和换底模式
"""

from pathlib import Path

import cv2

from pyhivision import IDPhotoSDK, PhotoRequest, create_settings
from pyhivision.schemas.alignment import AlignmentParams

# 配置模型路径
settings = create_settings(
    matting_models_dir="~/.pyhivision/matting",
    detection_models_dir="~/.pyhivision/detection",
)

# 创建 SDK 实例
sdk = IDPhotoSDK.create(settings=settings)

# 创建输出目录
output_dir = Path("examples/output/12_face_alignment")
output_dir.mkdir(parents=True, exist_ok=True)

# 读取输入图像（假设图像中的人脸有一定倾斜）
image = cv2.imread("examples/input/input_1.jpg")

print("=" * 60)
print("示例 1: 基础人脸矫正（默认参数）")
print("=" * 60)

# 创建请求（使用默认矫正参数）
request = PhotoRequest(
    image=image,
    size=(413, 295),  # 一寸照尺寸
    background_color=(255, 0, 0),  # 蓝色背景
    matting_model="modnet_photographic",
    detection_model="mtcnn",
    # alignment_params 使用默认值：
    # - enable_alignment=True
    # - angle_threshold=2.0
)

# 处理图像
result = sdk.process_single(request)

# 保存结果
cv2.imwrite(str(output_dir / "alignment_default.jpg"), result.standard)
if result.hd is not None:
    cv2.imwrite(str(output_dir / "alignment_default_hd.jpg"), result.hd)

print("✓ 处理完成")
if result.face_info:
    print(f"  - 矫正后的人脸角度: {result.face_info.roll_angle:.2f}°")
    print(f"  - 人脸位置: ({result.face_info.x}, {result.face_info.y})")
print()

# ============================================================================

print("=" * 60)
print("示例 2: 自定义角度阈值")
print("=" * 60)

# 只有角度超过 5° 才触发矫正
request2 = PhotoRequest(
    image=image,
    size=(413, 295),
    background_color=(255, 255, 255),  # 白色背景
    alignment_params=AlignmentParams(
        enable_alignment=True,
        angle_threshold=5.0,  # 更高的阈值
    ),
)

result2 = sdk.process_single(request2)
cv2.imwrite(str(output_dir / "alignment_threshold5.jpg"), result2.standard)

print("✓ 处理完成（阈值=5.0°）")
if result2.face_info:
    print(f"  - 人脸角度: {result2.face_info.roll_angle:.2f}°")
print()

# ============================================================================

print("=" * 60)
print("示例 3: 禁用人脸矫正")
print("=" * 60)

# 完全禁用矫正，保持原始角度
request3 = PhotoRequest(
    image=image,
    size=(413, 295),
    background_color=(0, 0, 255),  # 红色背景
    alignment_params=AlignmentParams(
        enable_alignment=False,  # 禁用矫正
    ),
)

result3 = sdk.process_single(request3)
cv2.imwrite(str(output_dir / "alignment_disabled.jpg"), result3.standard)

print("✓ 处理完成（矫正已禁用）")
if result3.face_info:
    print(f"  - 人脸角度（未矫正）: {result3.face_info.roll_angle:.2f}°")
print()

# ============================================================================

print("=" * 60)
print("示例 4: 换底模式 + 人脸矫正")
print("=" * 60)

# 在换底模式下也启用人脸矫正
request4 = PhotoRequest(
    image=image,
    size=(413, 295),
    change_bg_only=True,  # 换底模式
    alignment_params=AlignmentParams(
        enable_alignment=True,
        angle_threshold=2.0,
    ),
)

result4 = sdk.process_single(request4)
cv2.imwrite(str(output_dir / "alignment_change_bg_only.jpg"), result4.standard)

print("✓ 处理完成（换底模式）")
print("  - 矫正已应用")
print()

# ============================================================================

print("=" * 60)
print("示例 5: 对比矫正前后")
print("=" * 60)

# 处理两次：一次不矫正，一次矫正
request_before = PhotoRequest(
    image=image,
    size=(600, 400),
    background_color=(255, 255, 255),
    alignment_params=AlignmentParams(enable_alignment=False),
)

request_after = PhotoRequest(
    image=image,
    size=(600, 400),
    background_color=(255, 255, 255),
    alignment_params=AlignmentParams(enable_alignment=True, angle_threshold=1.0),
)

result_before = sdk.process_single(request_before)
result_after = sdk.process_single(request_after)

# 保存对比图
cv2.imwrite(str(output_dir / "before_alignment.jpg"), result_before.standard)
cv2.imwrite(str(output_dir / "after_alignment.jpg"), result_after.standard)

print("✓ 对比图已保存")
if result_before.face_info and result_after.face_info:
    print(f"  - 矫正前角度: {result_before.face_info.roll_angle:.2f}°")
    print(f"  - 矫正后角度: {result_after.face_info.roll_angle:.2f}°")
    print(f"  - 角度改善: {abs(result_before.face_info.roll_angle) - abs(result_after.face_info.roll_angle):.2f}°")
print()

# ============================================================================

# 清理资源
sdk.shutdown()

print("=" * 60)
print("所有示例执行完成！")
print("=" * 60)
print(f"\n输出文件位置：{output_dir}/")
print("  - alignment_default.jpg (默认参数)")
print("  - alignment_threshold5.jpg (阈值=5.0°)")
print("  - alignment_disabled.jpg (矫正禁用)")
print("  - alignment_change_bg_only.jpg (换底模式)")
print("  - before_alignment.jpg (矫正前)")
print("  - after_alignment.jpg (矫正后)")
