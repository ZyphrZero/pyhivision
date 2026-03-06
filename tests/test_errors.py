from pyhivision.exceptions.errors import FaceDetectionError


def test_face_detection_error_supports_optional_face_count():
    error = FaceDetectionError("validation failed")

    assert error.face_count is None
    assert str(error) == "validation failed"

