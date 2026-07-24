import importlib
import sys

import pytest

from pkasso.external.unipka.pka_predictor.api import (
    PredictionConfig,
    _run_inference,
)


def test_vendored_runtime_uses_only_package_qualified_names():
    pytest.importorskip("lmdb")
    modules_before = set(sys.modules)

    runtime = importlib.import_module("pkasso.external.unipka.unicoreinfer")
    unimol = importlib.import_module("pkasso.external.unipka.unimol")

    modules_added = set(sys.modules) - modules_before
    assert runtime.__name__ == "pkasso.external.unipka.unicoreinfer"
    assert unimol.__name__ == "pkasso.external.unipka.unimol"
    assert "unicoreinfer" not in modules_added
    assert "unimol" not in modules_added


def test_legacy_predictor_calls_vendored_inference_directly(monkeypatch, tmp_path):
    infer = pytest.importorskip("pkasso.external.unipka.unimol.infer")
    model_dir = tmp_path / "models"
    checkpoint = model_dir / "fold_0" / "checkpoint_best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    captured = {}

    def mock_run_inference(input_args):
        captured["input_args"] = input_args

    monkeypatch.setattr(infer, "run_inference", mock_run_inference)
    monkeypatch.setattr(
        "pkasso.external.unipka.pka_predictor.api.torch.cuda.is_available",
        lambda: False,
    )

    _run_inference(
        tmp_path / "processed",
        tmp_path / "results",
        "example",
        PredictionConfig(model_dir=model_dir, verbose=False),
    )

    input_args = captured["input_args"]
    assert input_args[0] == str(tmp_path / "processed")
    assert "--cpu" in input_args
    assert "--user-dir" not in input_args
