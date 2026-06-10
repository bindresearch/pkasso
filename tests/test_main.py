import importlib.util
import sys
import types
from pathlib import Path

import pytest
import numpy as np

from rdkit import Chem


def load_main_module():
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("pkasso")
    package.__path__ = [str(root / "pkasso")]

    predict_pka = types.ModuleType("pkasso.predict_pka")
    predict_pka.Predictor = object
    predict_pka.MolgpkaPredictor = object

    postprocess = types.ModuleType("pkasso.postprocess")
    postprocess.Molecule = type("Molecule", (), {})
    postprocess.Scan = type("Scan", (), {"__init__": lambda self, *args, **kwargs: None})
    postprocess.combine_results = lambda *args, **kwargs: None

    transitions = types.ModuleType("pkasso.transitions")
    transitions.calc_freqs_from_states = lambda *args, **kwargs: None
    transitions.calc_state_diffs = lambda *args, **kwargs: None

    coupling = types.ModuleType("pkasso.coupling")
    coupling.compare_pkas = lambda *args, **kwargs: None
    coupling.find_coupled_sites = lambda *args, **kwargs: []

    old_modules = {
        name: sys.modules.get(name)
        for name in (
            "pkasso",
            "pkasso.coupling",
            "pkasso.main",
            "pkasso.predict_pka",
            "pkasso.postprocess",
            "pkasso.transitions",
        )
    }

    sys.modules["pkasso"] = package
    sys.modules["pkasso.coupling"] = coupling
    sys.modules["pkasso.predict_pka"] = predict_pka
    sys.modules["pkasso.postprocess"] = postprocess
    sys.modules["pkasso.transitions"] = transitions

    spec = importlib.util.spec_from_file_location(
        "pkasso.main",
        root / "pkasso" / "main.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pkasso.main"] = module
    spec.loader.exec_module(module)

    for name, old_module in old_modules.items():
        if old_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old_module

    return module


main = load_main_module()

# @pytest.mark.parametrize(
#     ("smiles_raw","net_charge"),
#     [
#         (r"CCCCC(=O)O", 0),
#         (r"CCCCC(=O)[O-]", 0),
#         (r"[NH3+]CCCCC", 0),
#         (r"[N+](C)(C)(C)(C)", 1),
#         (r"CCCCC", 0),
#     ],
# )
# def test_preprocess(smiles_raw,net_charge):

#     mol, exclude_base_indices, exclude_acid_indices, phosphate_ohs = main.preprocess(smiles_raw,verbose=False)
#     qs = [at.GetFormalCharge() for at in mol.GetAtoms()]
#     print(qs)
#     for at_idx, q in enumerate(qs):
#         print(at_idx, q)
#         if q != 0:
#             assert (at_idx in exclude_base_indices) and (at_idx in exclude_acid_indices)
#     assert Chem.GetFormalCharge(mol) == net_charge

def test_find_candidate_sites():
    base = {
        0: 2.0,
        1: 7.0,
        2: 12.0,}
    acid = {
        0: 4.0,
        3: 12.0
    }
    exclude_base_indices = []
    exclude_acid_indices = []
    charged_indices = []
    indices, q_options = main.find_candidate_sites(
        base, acid, exclude_base_indices, exclude_acid_indices, charged_indices)
    expected_indices = [0, 1, 2, 3]
    expected_q_options = np.array([
        [1, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [1, 1, 0],
    ])
    assert (np.allclose(indices,expected_indices)) and (np.allclose(q_options,expected_q_options))


def test_find_candidate_sites_respects_excluded_and_charged_indices():
    base = {
        0: 2.0,
        1: 7.0,
        2: 12.0,
    }
    acid = {
        0: 4.0,
        3: 12.0,
    }
    indices, q_options = main.find_candidate_sites(
        base,
        acid,
        exclude_base_indices=[1],
        exclude_acid_indices=[0],
        charged_indices=[2],
    )
    expected_indices = [0, 1, 3]
    expected_q_options = np.array([
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
    ])
    assert (np.allclose(indices, expected_indices)) and (np.allclose(q_options, expected_q_options))

def test_construct_state_vectors():
    q_options = np.array([
        [1,0],
        [1,1],
        [0,1]
    ]).T
    cutoff_states = 100
    state_vecs = main.construct_state_vectors(q_options, cutoff_states)
    print(state_vecs)
    assert np.allclose(state_vecs, np.array([
        [0, 1],
        [0, 2],
        [1, 1],
        [1, 2]
    ])
    )
    cutoff_states = 2
    state_vecs = main.construct_state_vectors(q_options, cutoff_states)
    assert state_vecs == []
    # assert False

@pytest.mark.parametrize(
    ("smiles_raw",'net_charge'),
    [
        (r"NCCCCC", 1),
        (r"NCCCCCN", 2),
        (r"CCCCCO", -1),
        (r"NCCCCCO", 0),
        (r"Nc1ccc(O)cc1", 0),
        (r"Nc1ccc(N)cc1", 2),
    ],
)
def test_construct_mol(smiles_raw,net_charge):
    mol = Chem.MolFromSmiles(smiles_raw)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    indices = []
    state_vec = []

    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        map_idx = atom.GetAtomMapNum()
        if s == 'N':
            indices.append(map_idx)
            state_vec.append(2)
        elif s == 'O':
            indices.append(map_idx)
            state_vec.append(0)
    state_vec = np.array(state_vec)
    mol_cand = main.construct_mol(mol, indices, state_vec)
    assert Chem.GetFormalCharge(mol_cand) == net_charge
