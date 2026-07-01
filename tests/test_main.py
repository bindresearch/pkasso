import importlib.util
import itertools
import sys
import types
from pathlib import Path

import networkx as nx
import pytest
import numpy as np

from rdkit import Chem


def load_main_module():
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("pkasso")
    package.__path__ = [str(root / "pkasso")]

    predict_pka = types.ModuleType("pkasso.predict_pka")
    class Predictor:
        thermodynamic_prediction = "pka"

    class MolgpkaPredictor(Predictor):
        pass

    predict_pka.Predictor = Predictor
    predict_pka.MolgpkaPredictor = MolgpkaPredictor
    predict_pka.ThermodynamicPredictionMode = str

    postprocess = types.ModuleType("pkasso.postprocess")
    postprocess.Molecule = type("Molecule", (), {})
    postprocess.Scan = type("Scan", (), {"__init__": lambda self, *args, **kwargs: None})
    postprocess.combine_results = lambda *args, **kwargs: None

    transitions = types.ModuleType("pkasso.transitions")
    transitions.calc_freqs_from_states = lambda *args, **kwargs: None
    transitions.calc_state_diffs = lambda *args, **kwargs: None
    transitions.calc_state_pH_dependent_free_energies = (
        lambda standard_free_energies, state_vecs, pH, target_mean:
        np.asarray(standard_free_energies, dtype=np.float64)
        + np.array([np.sum(state_vec - 1) for state_vec in state_vecs]) * np.log(10) * (pH - target_mean)
    )
    transitions.calc_populations = lambda Gs: np.exp(-Gs) / np.sum(np.exp(-Gs))

    coupling = types.ModuleType("pkasso.coupling")
    coupling.compare_pkas = lambda *args, **kwargs: None
    coupling.compare_free_energies = lambda indices, *args, **kwargs: np.zeros(len(indices), dtype=np.float64)
    coupling.construct_free_energy_coupling_weight_matrix = (
        lambda indices, *args, **kwargs: np.zeros((len(indices), len(indices)), dtype=np.float64)
    )
    coupling.construct_coupling_weight_matrix = (
        lambda indices, *args, **kwargs: np.zeros((len(indices), len(indices)), dtype=np.float64)
    )
    coupling.find_coupled_sites = lambda *args, **kwargs: []
    def construct_state_vectors_single(indices, q_options):
        state_vecs = [np.ones(len(indices), dtype=np.int64)]
        for rel_idx, qs in enumerate(q_options):
            for q in (0, 2):
                if qs[q] == 1:
                    state_vec = np.ones(len(indices), dtype=np.int64)
                    state_vec[rel_idx] = q
                    state_vecs.append(state_vec)
        return state_vecs

    def construct_state_vectors_coupling(indices, q_options):
        state_vecs = construct_state_vectors_single(indices, q_options)
        state_vecs_by_str = {"".join(str(int(q)) for q in state_vec): state_vec for state_vec in state_vecs}
        for state_vec0, state_vec1 in itertools.combinations(state_vecs[1:], 2):
            changed0 = np.where(state_vec0 != 1)[0]
            changed1 = np.where(state_vec1 != 1)[0]
            if len(changed0) == 1 and len(changed1) == 1 and changed0[0] != changed1[0]:
                state_vec = np.ones(len(indices), dtype=np.int64)
                state_vec[changed0[0]] = state_vec0[changed0[0]]
                state_vec[changed1[0]] = state_vec1[changed1[0]]
                state_vecs_by_str.setdefault("".join(str(int(q)) for q in state_vec), state_vec)
        return list(state_vecs_by_str.values())

    coupling.construct_state_vectors_single = construct_state_vectors_single
    coupling.construct_state_vectors_coupling = construct_state_vectors_coupling

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
    base = [0, 1, 2]
    acid = [0, 3]
    exclude_base_indices = []
    exclude_acid_indices = []
    charged_indices = []
    indices, q_options = main.find_candidate_sites(base, acid, exclude_base_indices, exclude_acid_indices, charged_indices)
    expected_indices = [0, 1, 2, 3]
    expected_q_options = np.array(
        [
            [1, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 0],
        ]
    )
    assert (np.allclose(indices, expected_indices)) and (np.allclose(q_options, expected_q_options))


def test_find_candidate_sites_respects_excluded_and_charged_indices():
    base = [0, 1, 2]
    acid = [0, 3]
    indices, q_options = main.find_candidate_sites(
        base,
        acid,
        exclude_base_indices=[1],
        exclude_acid_indices=[0],
        charged_indices=[2],
    )
    expected_indices = [0, 1, 3]
    expected_q_options = np.array(
        [
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
        ]
    )
    assert (np.allclose(indices, expected_indices)) and (np.allclose(q_options, expected_q_options))


def test_construct_state_vectors():
    q_options = np.array([[1, 0], [1, 1], [0, 1]]).T
    cutoff_states = 100
    state_vecs = main.construct_state_vectors(q_options, cutoff_states)
    print(state_vecs)
    assert np.allclose(state_vecs, np.array([[0, 1], [0, 2], [1, 1], [1, 2]]))
    cutoff_states = 2
    state_vecs = main.construct_state_vectors(q_options, cutoff_states)
    assert state_vecs == []
    # assert False


def test_count_state_combinations():
    q_options = np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
        ]
    )
    assert main.count_state_combinations(q_options) == 6


def test_process_cluster_uses_batched_standard_free_energies_for_unipka_path():
    mol = Chem.MolFromSmiles("N")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    calls = []

    def predict_standard_free_energies(mols):
        calls.append([mol.GetProp("_Name") for mol in mols])
        return [0.0 for _ in mols]

    pk = main.pKasso(
        "N",
        tautomer_search=False,
        standard_free_energy_predictor=predict_standard_free_energies,
        standard_free_energy_config=types.SimpleNamespace(target_mean=6.0),
    )
    pk.mol0 = mol
    space = main.ProtonationIndexSpace(
        indices=[1],
        q_options=np.array([[1, 1, 1]], dtype=np.int64),
    )

    dist = pk.process_cluster(space, pH=7.0, sfreq_cutoff_individual=0.0, max_states_individual=10)
    freqs_by_state = dict(zip(dist.state_strs, dist.state_freqs))

    assert calls == [["0", "1", "2"]]
    assert freqs_by_state["0"] > freqs_by_state["1"] > freqs_by_state["2"]
    assert np.sum(dist.state_freqs) == pytest.approx(1.0)


def test_process_cluster_can_use_standard_free_energy_predictor_class():
    mol = Chem.MolFromSmiles("N")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    calls = []

    class StandardFreeEnergyPredictor(main.Predictor):
        thermodynamic_prediction = "standard_free_energy"

        @classmethod
        def predict_standard_free_energies(cls, mols, *, config=None):
            calls.append(([mol.GetProp("_Name") for mol in mols], config.target_mean))
            return [0.0 for _ in mols]

    pk = main.pKasso(
        "N",
        tautomer_search=False,
        pka_predictor_cls=StandardFreeEnergyPredictor,
        standard_free_energy_config=types.SimpleNamespace(target_mean=6.0),
    )
    pk.mol0 = mol
    space = main.ProtonationIndexSpace(
        indices=[1],
        q_options=np.array([[1, 1, 1]], dtype=np.int64),
    )

    pk.process_cluster(space, pH=7.0, sfreq_cutoff_individual=0.0, max_states_individual=10)

    assert calls == [(["0", "1", "2"], 6.0)]


def test_coupling_assay_weights_batches_double_states_for_unipka_path():
    mol = Chem.MolFromSmiles("NCCN")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    calls = []

    def predict_standard_free_energies(mols):
        calls.append([mol.GetProp("_Name") for mol in mols])
        return [0.0 for _ in mols]

    pk = main.pKasso(
        "NCCN",
        tautomer_search=False,
        standard_free_energy_predictor=predict_standard_free_energies,
    )
    pk.mol0 = mol
    pk.initialize_paths_models_libs()

    weights = pk.coupling_assay_weights(
        [1, 4],
        np.array(
            [
                [0, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.int64,
        ),
    )

    assert calls == [["11", "21", "12", "22"]]
    assert weights.shape == (2, 2)


def test_split_cluster_preserves_explicit_max_cut_edges_through_recursion(monkeypatch):
    q_options = np.ones((4, 3), dtype=np.int64)
    weights = np.zeros((4, 4), dtype=np.float64)
    seen_max_cut_edges = []

    def coupling_weights_to_graph(coupling_weights, coupling_cutoff, nodes=None):
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2, 3] if nodes is None else nodes)
        for edge in ((0, 1), (2, 3)):
            if all(node in graph for node in edge):
                graph.add_edge(*edge)
        return graph

    def find_best_penalty_limited_split(graph, coupling_weights, cluster_state_count, max_cut_edges, coupling_cutoff):
        seen_max_cut_edges.append(max_cut_edges)
        return []

    monkeypatch.setattr(main.coupling, "coupling_weights_to_graph", coupling_weights_to_graph, raising=False)
    monkeypatch.setattr(main.coupling, "find_best_penalty_limited_split", find_best_penalty_limited_split, raising=False)

    pk = main.pKasso("CC", cutoff_states=2, max_cut_edges=1)
    pk.split_cluster_by_coupling_penalty([0, 1, 2, 3], q_options, weights, 0.1, max_cut_edges=2)

    assert seen_max_cut_edges == [2, 2]


@pytest.mark.parametrize(
    ("smiles_raw", "net_charge"),
    [
        (r"NCCCCC", 1),
        (r"NCCCCCN", 2),
        (r"CCCCCO", -1),
        (r"NCCCCCO", 0),
        (r"Nc1ccc(O)cc1", 0),
        (r"Nc1ccc(N)cc1", 2),
    ],
)
def test_construct_mol(smiles_raw, net_charge):
    mol = Chem.MolFromSmiles(smiles_raw)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    indices = []
    state_vec = []

    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        map_idx = atom.GetAtomMapNum()
        if s == "N":
            indices.append(map_idx)
            state_vec.append(2)
        elif s == "O":
            indices.append(map_idx)
            state_vec.append(0)
    state_vec = np.array(state_vec)
    mol_cand = main.construct_mol(mol, indices, state_vec)
    assert Chem.GetFormalCharge(mol_cand) == net_charge
