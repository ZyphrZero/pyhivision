from types import SimpleNamespace

import pyhivision.cli as cli


def test_cmd_install_default_models_use_expected_types(monkeypatch):
    calls: list[tuple[str, str, str, bool]] = []
    monkeypatch.setattr(cli, "get_default_models_dir", lambda: "D:/models")

    def fake_download(model_name: str, model_type: str, models_dir: str, force: bool = False):
        calls.append((model_name, model_type, str(models_dir), force))
        return f"{models_dir}/{model_type}/{model_name}.onnx"

    monkeypatch.setattr(cli, "download_model", fake_download)

    args = SimpleNamespace(models_dir=None, all=False, model=None, force=False)
    exit_code = cli.cmd_install(args)

    assert exit_code == 0
    assert ("retinaface", "detection", "D:/models", False) in calls
    assert ("modnet_photographic", "matting", "D:/models", False) in calls
    assert ("hivision_modnet", "matting", "D:/models", False) in calls

