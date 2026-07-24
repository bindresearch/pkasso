import importlib.util
import itertools
import logging
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
        model_key = "predictor"
        thermodynamic_prediction = "pka"
        standard_free_energy_target_mean = None

    class MolgpkaPredictor(Predictor):
        model_key = "molgpka"

    class ResolvedPredictor:
        def __init__(self, predictor_cls, config):
            self.predictor_cls = predictor_cls
            self.config = config

    def resolve_models(model):
        if model != {"molgpka": {}}:
            raise ValueError(f"Unsupported test model: {model}")
        return (ResolvedPredictor(MolgpkaPredictor, None),)

    predict_pka.ModelInput = dict
    predict_pka.Predictor = Predictor
    predict_pka.MolgpkaPredictor = MolgpkaPredictor
    predict_pka.ResolvedPredictor = ResolvedPredictor
    predict_pka.ThermodynamicPredictionMode = str
    predict_pka.resolve_models = resolve_models

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


def test_screen_clusters_skips_coupling_when_full_state_space_fits(monkeypatch):
    q_options = np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    pk = main.pKasso("CC", cutoff_states=6)

    def coupling_assay_weights(*args, **kwargs):
        raise AssertionError("coupling assay should be skipped")

    monkeypatch.setattr(pk, "coupling_assay_weights", coupling_assay_weights)

    assert pk.screen_clusters([1, 2], q_options) == [[0, 1]]


def test_screen_clusters_runs_coupling_when_full_state_space_exceeds_cutoff(monkeypatch):
    q_options = np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    pk = main.pKasso("CC", cutoff_states=5)
    calls = []

    def coupling_assay_weights(indices, q_options, context=None):
        calls.append((indices, q_options.copy(), context))
        return np.zeros((len(indices), len(indices)), dtype=np.float64)

    def coupling_weights_to_graph(coupling_weights, coupling_cutoff, nodes=None):
        graph = nx.Graph()
        graph.add_nodes_from(range(coupling_weights.shape[0]) if nodes is None else nodes)
        return graph

    monkeypatch.setattr(pk, "coupling_assay_weights", coupling_assay_weights)
    monkeypatch.setattr(main.coupling, "coupling_weights_to_graph", coupling_weights_to_graph, raising=False)

    assert pk.screen_clusters([1, 2], q_options) == [[0], [1]]
    assert len(calls) == 1
    assert calls[0][0] == [1, 2]
    assert np.array_equal(calls[0][1], q_options)


def test_screen_clusters_decouples_phosphate_groups_after_coupling(monkeypatch):
    mol = Chem.MolFromSmiles("CCOP(=O)(O)O")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    _, phosphate_groups = main.special_cases.has_phosphate(mol)
    phosphate_ohs = next(iter(phosphate_groups.values()))
    indices = [1, *phosphate_ohs]
    q_options = np.tile(np.array([[1, 1, 0]], dtype=np.int64), (len(indices), 1))

    pk = main.pKasso("CCOP(=O)(O)O", cutoff_states=100)
    pk.mol0 = mol
    calls = []

    def coupling_assay_weights(indices_arg, q_options_arg, context=None):
        calls.append((indices_arg, q_options_arg.copy(), context))
        weights = np.ones((len(indices_arg), len(indices_arg)), dtype=np.float64)
        np.fill_diagonal(weights, 0.0)
        return weights

    def coupling_weights_to_graph(coupling_weights, coupling_cutoff, nodes=None):
        graph = nx.Graph()
        graph.add_nodes_from(range(coupling_weights.shape[0]) if nodes is None else nodes)
        for idx, i in enumerate(graph.nodes):
            for j in list(graph.nodes)[idx + 1:]:
                if max(coupling_weights[i, j], coupling_weights[j, i]) >= coupling_cutoff:
                    graph.add_edge(i, j)
        return graph

    monkeypatch.setattr(pk, "coupling_assay_weights", coupling_assay_weights)
    monkeypatch.setattr(main.coupling, "coupling_weights_to_graph", coupling_weights_to_graph, raising=False)

    assert pk.screen_clusters(indices, q_options) == [[0], list(range(1, len(indices)))]
    assert len(calls) == 1


def test_screen_clusters_does_not_decouple_phosphates_for_non_molgpka_context(monkeypatch):
    class OtherPredictor(main.Predictor):
        pass

    mol = Chem.MolFromSmiles("CCOP(=O)(O)O")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    _, phosphate_groups = main.special_cases.has_phosphate(mol)
    phosphate_ohs = next(iter(phosphate_groups.values()))
    indices = [1, *phosphate_ohs]
    q_options = np.tile(np.array([[1, 1, 0]], dtype=np.int64), (len(indices), 1))

    pk = main.pKasso("CCOP(=O)(O)O", cutoff_states=100)
    pk.mol0 = mol
    context = main.PredictorContext(predictor_cls=OtherPredictor)

    def coupling_assay_weights(*args, **kwargs):
        raise AssertionError("coupling assay should be skipped")

    monkeypatch.setattr(pk, "coupling_assay_weights", coupling_assay_weights)

    assert pk.screen_clusters(indices, q_options, context=context) == [list(range(len(indices)))]


def test_combine_expert_energies_aligns_on_population_weighted_shared_states():
    space = main.ProtonationIndexSpace(
        indices=[1, 2],
        q_options=np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int64),
    )
    raw_a = main.RawMicrostateEnergies(
        index_space=space,
        pH=7.0,
        state_strs=["00", "01", "11"],
        state_vecs=[np.array([0, 0]), np.array([0, 1]), np.array([1, 1])],
        Gs=np.array([0.0, 3.0, 2.0]),
    )
    raw_b = main.RawMicrostateEnergies(
        index_space=space,
        pH=7.0,
        state_strs=["10", "11", "00"],
        state_vecs=[np.array([1, 0]), np.array([1, 1]), np.array([0, 0])],
        Gs=np.array([3.0, 5.0, 1.0]),
    )

    combined = main.combine_expert_energies([raw_a, raw_b])

    assert combined.state_strs == ["00", "01", "10", "11"]
    assert combined.Gs.tolist() == pytest.approx([0.0, 3.04742587, 1.95257413, 3.0])
    assert combined.Gs_sigmas.tolist() == pytest.approx([0.06707031, 2.0, 2.0, 1.34714325])


def test_combine_expert_energies_falls_back_to_first_model_without_shared_state(caplog):
    space = main.ProtonationIndexSpace(
        indices=[1],
        q_options=np.array([[1, 1, 0]], dtype=np.int64),
    )
    raw_a = main.RawMicrostateEnergies(
        index_space=space,
        pH=7.0,
        state_strs=["0"],
        state_vecs=[np.array([0])],
        Gs=np.array([0.0]),
    )
    raw_b = main.RawMicrostateEnergies(
        index_space=space,
        pH=7.0,
        state_strs=["1"],
        state_vecs=[np.array([1])],
        Gs=np.array([0.0]),
    )

    with caplog.at_level(logging.INFO, logger=main.logger.name):
        combined = main.combine_expert_energies([raw_a, raw_b])

    assert "Falling back to the first model provided" in caplog.text
    assert combined.index_space is space
    assert combined.state_strs == ["0"]
    assert combined.state_vecs[0].tolist() == [0]
    assert combined.Gs.tolist() == pytest.approx([0.0])
    assert combined.Gs_sigmas.tolist() == pytest.approx([2.0])


def test_combine_expert_energies_pads_different_index_spaces_with_neutral_state():
    space_a = main.ProtonationIndexSpace(
        indices=[1],
        q_options=np.array([[1, 1, 0]], dtype=np.int64),
    )
    space_b = main.ProtonationIndexSpace(
        indices=[2],
        q_options=np.array([[1, 1, 0]], dtype=np.int64),
    )
    raw_a = main.RawMicrostateEnergies(
        index_space=space_a,
        pH=7.0,
        state_strs=["0", "1"],
        state_vecs=[np.array([0]), np.array([1])],
        Gs=np.array([2.0, 0.0]),
    )
    raw_b = main.RawMicrostateEnergies(
        index_space=space_b,
        pH=7.0,
        state_strs=["0", "1"],
        state_vecs=[np.array([0]), np.array([1])],
        Gs=np.array([3.0, 1.0]),
    )

    combined = main.combine_expert_energies([raw_a, raw_b])

    assert combined.index_space.indices == [1, 2]
    assert combined.index_space.q_options.tolist() == [[1, 1, 0], [1, 1, 0]]
    assert combined.state_strs == ["01", "10", "11"]
    assert combined.Gs.tolist() == pytest.approx([2.0, 2.0, 0.0])


def test_bind_combined_context_space_uses_union_of_predictor_indices():
    pk = main.pKasso("CC", tautomer_search=False)
    context_a = main.PredictorContext(predictor_cls=main.MolgpkaPredictor)
    context_a.indices0 = [2]
    context_a.q_options0 = np.array([[1, 1, 0]], dtype=np.int64)
    context_a.clusters = [[0]]
    context_b = main.PredictorContext(predictor_cls=main.MolgpkaPredictor)
    context_b.indices0 = [1]
    context_b.q_options0 = np.array([[0, 1, 1]], dtype=np.int64)
    context_b.clusters = [[0]]
    pk.predictor_contexts = [context_a, context_b]
    pk.primary_context = context_a

    pk._bind_combined_context_space()

    assert pk.indices0 == [1, 2]
    assert pk.q_options0.tolist() == [[0, 1, 1], [1, 1, 0]]
    assert pk.index_space0.indices == [1, 2]


def test_process_cluster_uses_batched_standard_free_energies_for_unipka_path():
    mol = Chem.MolFromSmiles("N")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    calls = []

    class StandardFreeEnergyPredictor(main.Predictor):
        thermodynamic_prediction = "standard_free_energy"
        standard_free_energy_target_mean = 6.0

        @classmethod
        def predict_standard_free_energies(cls, mols, *, config=None):
            calls.append([mol.GetProp("_Name") for mol in mols])
            return [0.0 for _ in mols]

    pk = main.pKasso(
        "N",
        tautomer_search=False,
    )
    pk.resolved_predictors = (
        main.ResolvedPredictor(StandardFreeEnergyPredictor, None),
    )
    pk.mol0 = mol
    space = main.ProtonationIndexSpace(
        indices=[1],
        q_options=np.array([[1, 1, 1]], dtype=np.int64),
    )

    raw = pk.process_cluster(space, pH=7.0, free_energy_cutoff_individual=np.inf, max_states_individual=10)
    dist = pk._finalize_microstates(raw)
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
        standard_free_energy_target_mean = 6.0

        @classmethod
        def predict_standard_free_energies(cls, mols, *, config=None):
            calls.append(([mol.GetProp("_Name") for mol in mols], config.target_mean))
            return [0.0 for _ in mols]

    pk = main.pKasso(
        "N",
        tautomer_search=False,
    )
    pk.resolved_predictors = (
        main.ResolvedPredictor(
            StandardFreeEnergyPredictor,
            types.SimpleNamespace(target_mean=6.0),
        ),
    )
    pk.mol0 = mol
    space = main.ProtonationIndexSpace(
        indices=[1],
        q_options=np.array([[1, 1, 1]], dtype=np.int64),
    )

    pk.process_cluster(space, pH=7.0, free_energy_cutoff_individual=np.inf, max_states_individual=10)

    assert calls == [(["0", "1", "2"], 6.0)]


def test_coupling_assay_weights_batches_double_states_for_unipka_path():
    mol = Chem.MolFromSmiles("NCCN")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    calls = []

    class StandardFreeEnergyPredictor(main.Predictor):
        thermodynamic_prediction = "standard_free_energy"
        standard_free_energy_target_mean = 6.457855284082695

        @classmethod
        def predict_standard_free_energies(cls, mols, *, config=None):
            calls.append([mol.GetProp("_Name") for mol in mols])
            return [0.0 for _ in mols]

    pk = main.pKasso(
        "NCCN",
        tautomer_search=False,
    )
    pk.resolved_predictors = (
        main.ResolvedPredictor(StandardFreeEnergyPredictor, None),
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
