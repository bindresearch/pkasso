from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))
pytest.importorskip("lmdb")

from unicoreinfer import checkpoint_utils, options, tasks, utils  # noqa: E402
from unicoreinfer.data import Dictionary, NestedDictionaryDataset  # noqa: E402
from unicoreinfer.losses import UnicoreLoss  # noqa: E402
from unicoreinfer.models import BaseUnicoreModel  # noqa: E402
from unicoreinfer.modules import LayerNorm, TransformerEncoderLayer  # noqa: E402


def test_runtime_uses_an_isolated_import_namespace():
    import unicoreinfer

    assert unicoreinfer.__name__ == "unicoreinfer"
    assert not (RUNTIME_ROOT / "unicore").exists()


def test_inference_runtime_surface_is_available():
    assert callable(checkpoint_utils.load_checkpoint_to_cpu)
    assert callable(options.get_validation_parser)
    assert callable(tasks.setup_task)
    assert callable(utils.move_to_cuda)
    assert Dictionary is not None
    assert NestedDictionaryDataset is not None
    assert UnicoreLoss is not None
    assert BaseUnicoreModel is not None
    assert TransformerEncoderLayer is not None


def test_training_runtime_is_not_shipped():
    assert importlib.util.find_spec("unicoreinfer.optim") is None
    assert importlib.util.find_spec("unicoreinfer.trainer") is None
    assert importlib.util.find_spec("unicoreinfer.ema") is None


def test_layer_norm_uses_pytorch_fallback_on_cpu():
    layer = LayerNorm(8).eval()
    values = torch.randn(2, 4, 8)

    result = layer(values)

    assert result.shape == values.shape
    assert result.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_inference_primitives_without_fused_extensions():
    layer = LayerNorm(8).eval().half().cuda()
    values = torch.randn(2, 4, 8).half()
    sample = {"values": values, "metadata": ["unchanged"]}

    moved = utils.move_to_cuda(sample)
    result = layer(moved["values"])

    assert moved["values"].is_cuda
    assert result.is_cuda
    assert result.dtype == torch.float16
    assert moved["metadata"] == ["unchanged"]
