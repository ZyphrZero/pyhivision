#!/usr/bin/env python
"""错误处理示例 - 处理各种异常情况"""

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


def main():
    settings = create_settings(
        matting_models_dir="~/.pyhivision/models/matting",
    )
    sdk = IDPhotoSDK.create(settings=settings)

    try:
        image = cv2.imread("./examples/input/input_1.jpg")
        if image is None:
            raise FileNotFoundError("无法读取图像文件")

        request = PhotoRequest(
            image=image,
            size=(413, 295),
            background_color=(255, 0, 0),
            matting_model="hivision_modnet",
            # matting_model="rmbg_1_4",  # 故意使用不支持的模型以触发错误
        )

        result = sdk.process_single(request)
        cv2.imwrite("output.jpg", result.standard)

        print("✅ 处理成功")

    except ValidationError as e:
        print(f"❌ 数据验证失败: {e}")
    except NoFaceDetectedError:
        print("❌ 未检测到人脸，请确保照片中有清晰的人脸")
    except FaceDetectionError as e:
        print(f"❌ 人脸检测失败: {e}")
    except MattingError as e:
        print(f"❌ 抠图失败: {e}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")


if __name__ == "__main__":
    main()
