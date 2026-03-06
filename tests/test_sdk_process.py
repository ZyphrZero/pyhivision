import numpy as np
import pytest

from pyhivision.core.pipeline import IDPhotoSDK


class _DummyPipeline:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False

    def process_single(self, request):  # pragma: no cover - runtime behavior only
        raise RuntimeError("boom")


def test_sdk_process_uses_context_manager_for_cleanup(monkeypatch):
    dummy = _DummyPipeline()
    monkeypatch.setattr(IDPhotoSDK, "create", staticmethod(lambda settings=None: dummy))

    with pytest.raises(RuntimeError, match="boom"):
        IDPhotoSDK.process(image=np.zeros((100, 100, 3), dtype=np.uint8), size=(413, 295))

    assert dummy.entered is True
    assert dummy.exited is True

