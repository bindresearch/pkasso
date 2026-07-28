import inspect
import sys

import pandas as pd
import pytest
from rdkit import Chem

import pkasso.predict_pka as predict_pka
from pkasso.predict_pka import (
    MolgpkaPredictor,
    UnipkaPredictor,
    resolve_models,
    resolve_predictor_cls,
)

try:
    import unipkainfer
except ModuleNotFoundError as exc:
    if exc.name != "unipkainfer":
        raise
    unipkainfer = None

requires_unipka = pytest.mark.skipif(
    unipkainfer is None,
    reason="requires the pkasso[unipka] optional dependency",
)


def mapped_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mol


def test_predictor_apis_do_not_expose_device_keyword():
    assert "device" not in inspect.signature(predict_pka.Predictor).parameters
    assert "device" not in inspect.signature(predict_pka.MolgpkaPredictor).parameters
    assert "device" not in inspect.signature(predict_pka.UnipkaPredictor).parameters
    assert "device" not in inspect.signature(predict_pka.predict_acid).parameters
    assert "device" not in inspect.signature(predict_pka.predict_base).parameters


def test_unipka_predictor_carboxylic_acid_site_ids():
    predictor = UnipkaPredictor(mapped_mol("CCCCC(=O)O"))

    assert predictor.pred_acid_ids() == [7]
    assert predictor.pred_base_ids() == []
    assert predictor.exclude_sites() == ([], [])
    assert predictor.pred_acid() == {}
    assert predictor.pred_base() == {}
    assert not hasattr(predictor, "model")


def test_unipka_predictor_carboxylate_base_site_ids():
    predictor = UnipkaPredictor(mapped_mol("CCCCC(=O)[O-]"))

    assert predictor.pred_acid_ids() == []
    assert predictor.pred_base_ids() == [7]


def test_unipka_predictor_amine_site_ids():
    predictor = UnipkaPredictor(mapped_mol("NCCCCC"))

    assert predictor.pred_acid_ids() == []
    assert predictor.pred_base_ids() == [1]


def test_unipka_predictor_ammonium_site_ids():
    predictor = UnipkaPredictor(mapped_mol("[NH3+]CCCCC"))

    assert predictor.pred_acid_ids() == [1]
    assert predictor.pred_base_ids() == []


def test_unipka_predictor_excludes_poly_aza_ring_base_sites():
    predictor = UnipkaPredictor(mapped_mol("Fc1ccc2c(C=Cc3nnn[nH]3)c[nH]c2c1"))

    assert predictor.pred_base_ids() == [10, 11, 12]
    assert predictor.exclude_sites() == ([10, 11, 12, 13, 15], [])


def test_molgpka_close_cation_penalty_uses_target_net_charge(monkeypatch):
    mol = mapped_mol("NC(Cc1c[nH]cn1)C(=O)O")
    predictor = MolgpkaPredictor(mol)
    target_map_idx = next(
        atom.GetAtomMapNum()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N"
        and atom.GetIsAromatic()
        and atom.GetTotalNumHs() == 0
    )

    monkeypatch.setattr(
        predictor,
        "_predict_base_raw",
        lambda: {target_map_idx: 6.0},
    )
    monkeypatch.setattr(
        predict_pka,
        "has_nplus_base_proximity",
        lambda *args, **kwargs: 1,
    )

    predictor.source_net_charge = 0
    assert predictor.pred_base()[target_map_idx] == pytest.approx(6.0)

    predictor.source_net_charge = 1
    assert predictor.pred_base()[target_map_idx] == pytest.approx(3.5)


@requires_unipka
def test_unipka_predictor_curates_standard_free_energy_per_molecule(monkeypatch):
    class CuratingUnipkaPredictor(UnipkaPredictor):
        heavy_atom_counts: list[int] = []

        def _curate_free_energy(self, standard_free_energy: float) -> float:
            self.heavy_atom_counts.append(self.mol.GetNumHeavyAtoms())
            return standard_free_energy + self.mol.GetNumHeavyAtoms()

    def mock_predict_standard_free_energies(mols, *, config=None):
        assert config == "test-config"
        return pd.DataFrame(
            {
                "molecule_index": [0, 1],
                "standard_free_energy": [1.5, 2.5],
            }
        )

    monkeypatch.setattr(unipkainfer, "predict_standard_free_energies", mock_predict_standard_free_energies)

    results = CuratingUnipkaPredictor.predict_standard_free_energies(
        [mapped_mol("C"), mapped_mol("CC")],
        config="test-config",
    )

    assert results["standard_free_energy"].tolist() == [2.5, 4.5]
    assert CuratingUnipkaPredictor.heavy_atom_counts == [1, 2]


def test_resolve_predictor_cls_accepts_public_model_keys():
    assert resolve_predictor_cls("molgpka") is MolgpkaPredictor
    assert resolve_predictor_cls("unipka") is UnipkaPredictor
    assert UnipkaPredictor.standard_free_energy_target_mean == 6.457855284082695


@requires_unipka
def test_resolve_models_preserves_order_and_delegates_options():
    resolved = resolve_models(
        {
            "molgpka": {},
            "unipka": {
                "folds": (0, 1),
                "gpu": False,
            },
        },
        nthreads=4,
    )

    assert [item.predictor_cls for item in resolved] == [MolgpkaPredictor, UnipkaPredictor]
    assert resolved[0].config is None
    assert resolved[1].config == unipkainfer.UnipkaFreeEnergyConfig(
        folds=(0, 1),
        nthreads=4,
        gpu=False,
    )


@pytest.mark.parametrize("model", ["unipka", ["unipka"], (), {}])
def test_resolve_models_requires_nonempty_mapping(model):
    expected_error = ValueError if model == {} else TypeError
    with pytest.raises(expected_error):
        resolve_models(model)


def test_molgpka_rejects_model_options():
    with pytest.raises(ValueError, match="does not accept model options"):
        resolve_models({"molgpka": {"gpu": False}})


@requires_unipka
def test_unipka_rejects_unknown_model_options():
    with pytest.raises(ValueError, match="Unknown unipka option"):
        resolve_models({"unipka": {"batch_size": 4}})


@requires_unipka
def test_unipka_rejects_nested_nthreads_option():
    with pytest.raises(ValueError, match="top-level 'nthreads'"):
        resolve_models({"unipka": {"nthreads": 4}})


def test_resolve_models_rejects_negative_nthreads():
    with pytest.raises(ValueError, match="nthreads must be at least 0"):
        resolve_models({"molgpka": {}}, nthreads=-1)


def test_unipka_recommends_optional_extra_when_package_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "unipkainfer", None)

    with pytest.raises(ModuleNotFoundError, match=r"pip install 'pkasso\[unipka\]'"):
        resolve_models({"unipka": {}})
