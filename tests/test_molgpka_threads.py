import pytest

from pkasso.external.molgpka import pka


def test_configure_torch_threads_uses_requested_positive_count(monkeypatch):
    configured = []
    monkeypatch.setattr(pka.torch, "set_num_threads", configured.append)

    pka.configure_torch_threads(3)

    assert configured == [3]


def test_configure_torch_threads_restores_default_for_zero(monkeypatch):
    configured = []
    monkeypatch.setattr(pka.torch, "set_num_threads", configured.append)
    monkeypatch.setattr(pka, "_DEFAULT_TORCH_NUM_THREADS", 7)

    pka.configure_torch_threads(0)

    assert configured == [7]


def test_configure_torch_threads_rejects_negative_count():
    with pytest.raises(ValueError, match="nthreads must be at least 0"):
        pka.configure_torch_threads(-1)
