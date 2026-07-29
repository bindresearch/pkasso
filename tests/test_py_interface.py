import pytest
from rdkit import Chem

from pkasso import py_interface
from pkasso.predict_pka import UnipkaPredictor


class Molecule:
    smiles = ("C",)
    mols = ()


def test_protonate_passes_model_mapping_to_pkasso(monkeypatch):
    captured = {}

    class PKasso:
        def __init__(self, smiles, **kwargs):
            captured["smiles"] = smiles
            captured.update(kwargs)

        def run_single(self, pH):
            captured["pH"] = pH
            return Molecule()

    monkeypatch.setattr(py_interface, "pKasso", PKasso)
    model = {"unipka": {"folds": (0, 1), "gpu": False}}

    py_interface.protonate("C", pH=6.5, model=model)

    assert captured["smiles"] == "C"
    assert captured["pH"] == 6.5
    assert captured["model"] is model


def test_batch_protonate_passes_model_mapping_for_each_molecule(monkeypatch):
    captured = []

    class PKasso:
        def __init__(self, smiles, **kwargs):
            captured.append((smiles, kwargs))

        def run_single(self, pH):
            return Molecule()

    monkeypatch.setattr(py_interface, "pKasso", PKasso)
    model = {"molgpka": {}, "unipka": {}}

    py_interface.batch_protonate(["C", "N"], model=model)

    assert [smiles for smiles, _ in captured] == ["C", "N"]
    assert all(kwargs["model"] is model for _, kwargs in captured)


def test_batch_protonate_converts_rdkit_molecules_to_smiles(monkeypatch):
    captured = []

    class PKasso:
        def __init__(self, smiles, **kwargs):
            captured.append(smiles)

        def run_single(self, pH):
            return Molecule()

    monkeypatch.setattr(py_interface, "pKasso", PKasso)
    mol = Chem.MolFromSmiles("CCO")

    py_interface.batch_protonate([mol])

    assert captured == ["CCO"]


def test_scan_ph_passes_model_mapping_to_pkasso(monkeypatch):
    captured = {}

    class PKasso:
        def __init__(self, smiles, **kwargs):
            captured["smiles"] = smiles
            captured.update(kwargs)

        def run_scan(self, pHs):
            captured["pHs"] = pHs
            return "scan"

    monkeypatch.setattr(py_interface, "pKasso", PKasso)
    model = {"molgpka": {}, "unipka": {}}

    scan = py_interface.scan_pH(
        "C",
        pHs=[6.0, 7.0],
        model=model,
        nthreads=4,
        output_molecules_from_scan=False,
    )

    assert scan == "scan"
    assert captured["smiles"] == "C"
    assert captured["pHs"].tolist() == [6.0, 7.0]
    assert captured["model"] is model
    assert captured["nthreads"] == 4
    assert captured["output_molecules_from_scan"] is False


def test_default_model_is_left_to_pkasso(monkeypatch):
    captured = {}

    class PKasso:
        def __init__(self, smiles, **kwargs):
            captured.update(kwargs)

        def run_single(self, pH):
            return Molecule()

    monkeypatch.setattr(py_interface, "pKasso", PKasso)

    py_interface.protonate("C")

    assert "model" not in captured


def test_entry_points_reject_predictor_classes_kwargs():
    with pytest.raises(ValueError, match="model mapping"):
        py_interface.protonate(
            "C",
            model={"unipka": {}},
            pka_predictor_classes=(UnipkaPredictor,),
        )


def test_entry_points_reject_predictor_class_kwargs():
    with pytest.raises(ValueError, match="model mapping"):
        py_interface.protonate(
            "C",
            model={"unipka": {}},
            pka_predictor_cls=UnipkaPredictor,
        )
