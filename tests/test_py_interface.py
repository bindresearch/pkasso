import pytest

from pkasso import py_interface
from pkasso.predict_pka import MolgpkaPredictor, UnipkaPredictor


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


def test_protonate_resolves_model_key_list_to_predictor_classes(monkeypatch):
    captured = {}

    class PKasso:
        def __init__(self, smiles, **kwargs):
            captured["smiles"] = smiles
            captured.update(kwargs)

        def run_single(self, pH):
            captured["pH"] = pH
            return Molecule()

    monkeypatch.setattr(py_interface, "pKasso", PKasso)

    py_interface.protonate("C", pH=6.5, model=["molgpka", "unipka"])

    assert captured["smiles"] == "C"
    assert captured["pH"] == 6.5
    assert captured["pka_predictor_classes"] == (MolgpkaPredictor, UnipkaPredictor)
    assert "pka_predictor_cls" not in captured


def test_batch_protonate_resolves_model_key_list_to_predictor_classes(monkeypatch):
    captured = []

    class PKasso:
        def __init__(self, smiles, **kwargs):
            captured.append((smiles, kwargs))

        def run_single(self, pH):
            return Molecule()

    monkeypatch.setattr(py_interface, "pKasso", PKasso)

    py_interface.batch_protonate(["C", "N"], model=("molgpka", "unipka"))

    assert [smiles for smiles, _ in captured] == ["C", "N"]
    assert all(
        kwargs["pka_predictor_classes"] == (MolgpkaPredictor, UnipkaPredictor)
        for _, kwargs in captured
    )


def test_scan_ph_resolves_model_key_list_to_predictor_classes(monkeypatch):
    captured = {}

    class PKasso:
        def __init__(self, smiles, **kwargs):
            captured["smiles"] = smiles
            captured.update(kwargs)

        def run_scan(self, pHs):
            captured["pHs"] = pHs
            return "scan"

    monkeypatch.setattr(py_interface, "pKasso", PKasso)

    scan = py_interface.scan_pH("C", pHs=[6.0, 7.0], model=["molgpka", "unipka"])

    assert scan == "scan"
    assert captured["smiles"] == "C"
    assert captured["pHs"].tolist() == [6.0, 7.0]
    assert captured["pka_predictor_classes"] == (MolgpkaPredictor, UnipkaPredictor)


def test_single_item_model_list_uses_single_predictor_class(monkeypatch):
    captured = {}

    class PKasso:
        def __init__(self, smiles, **kwargs):
            captured.update(kwargs)

        def run_single(self, pH):
            return Molecule()

    monkeypatch.setattr(py_interface, "pKasso", PKasso)

    py_interface.protonate("C", model=["unipka"])

    assert captured["pka_predictor_cls"] is UnipkaPredictor
    assert "pka_predictor_classes" not in captured


def test_entry_points_reject_predictor_classes_kwargs():
    with pytest.raises(ValueError, match="entry points accept model keys"):
        py_interface.protonate("C", model="unipka", pka_predictor_classes=(UnipkaPredictor,))


def test_entry_points_reject_predictor_class_kwargs():
    with pytest.raises(ValueError, match="entry points accept model keys"):
        py_interface.protonate("C", model="unipka", pka_predictor_cls=UnipkaPredictor)
