import pytest

from pkasso import py_interface
from pkasso.predict_pka import UnipkaPredictor


class Molecule:
    smiles = ("C",)
    mols = ()


def test_protonate_resolves_model_key_to_predictor_class(monkeypatch):
    captured = {}

    class PKasso:
        def __init__(self, smiles, **kwargs):
            captured["smiles"] = smiles
            captured.update(kwargs)

        def run_single(self, pH):
            captured["pH"] = pH
            return Molecule()

    monkeypatch.setattr(py_interface, "pKasso", PKasso)

    py_interface.protonate("C", pH=6.5, model="unipka")

    assert captured["smiles"] == "C"
    assert captured["pH"] == 6.5
    assert captured["pka_predictor_cls"] is UnipkaPredictor


def test_model_key_conflicts_with_predictor_class():
    with pytest.raises(ValueError, match="Pass either model or pka_predictor_cls"):
        py_interface.protonate("C", model="unipka", pka_predictor_cls=UnipkaPredictor)
