import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest


def load_coupling_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "pkasso_coupling",
        root / "pkasso" / "coupling.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coupling = load_coupling_module()


def test_cluster_coupling_matrix_uses_connected_components():
    matrix = np.zeros((6, 6), dtype=np.int64)
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 5),
        (5, 3),
    ]
    for i, j in edges:
        matrix[i, j] = 1

    assert coupling.cluster_coupling_matrix(matrix) == [[0, 1, 2], [3, 4, 5]]


def test_coupling_weights_to_graph_thresholds_undirected_max_weights():
    weights = np.zeros((4, 4), dtype=np.float64)
    weights[0, 1] = 0.2
    weights[2, 1] = 0.4
    weights[2, 3] = 0.05

    graph = coupling.coupling_weights_to_graph(weights, coupling_cutoff=0.1)

    assert sorted(graph.edges()) == [(0, 1), (1, 2)]
    assert graph[0][1]["weight"] == pytest.approx(0.2)
    assert graph[1][2]["weight"] == pytest.approx(0.4)


def test_find_best_penalty_limited_split_prefers_lowest_state_count():
    weights = np.zeros((5, 5), dtype=np.float64)
    for i in range(4):
        weights[i, i + 1] = 0.2

    graph = coupling.coupling_weights_to_graph(weights, coupling_cutoff=0.1)

    split = coupling.find_best_penalty_limited_split(
        graph,
        weights,
        lambda cluster: 2 ** len(cluster),
        max_cut_edges=1,
        coupling_cutoff=0.15,
    )

    assert split == [[0, 1], [2, 3, 4]]


def test_find_best_penalty_limited_split_can_use_two_edge_cutsets():
    weights = np.zeros((5, 5), dtype=np.float64)
    for i in range(4):
        weights[i, i + 1] = 0.2

    graph = coupling.coupling_weights_to_graph(weights, coupling_cutoff=0.1)

    split = coupling.find_best_penalty_limited_split(
        graph,
        weights,
        lambda cluster: 2 ** len(cluster),
        max_cut_edges=2,
        coupling_cutoff=0.17,
    )

    assert split == [[0], [1, 2], [3, 4]]


def test_find_best_penalty_limited_split_makes_two_edge_cuts_harder():
    weights = np.zeros((4, 4), dtype=np.float64)
    weights[0, 1] = 0.16
    weights[1, 2] = 0.16
    weights[2, 3] = 0.16
    graph = coupling.coupling_weights_to_graph(weights, coupling_cutoff=0.1)

    split = coupling.find_best_penalty_limited_split(
        graph,
        weights,
        lambda cluster: 2 ** len(cluster),
        max_cut_edges=2,
        coupling_cutoff=0.1,
    )

    assert split == [[0, 1], [2, 3]]


def test_find_best_penalty_limited_split_rejects_expensive_cuts():
    graph = nx.path_graph(3)
    weights = np.zeros((3, 3), dtype=np.float64)
    weights[0, 1] = 0.2
    weights[1, 2] = 0.2

    split = coupling.find_best_penalty_limited_split(
        graph,
        weights,
        lambda cluster: 2 ** len(cluster),
        max_cut_edges=1,
        coupling_cutoff=0.1,
        cut_penalty_factor=1.0,
    )

    assert split is None


def test_cut_search_validation():
    graph = nx.path_graph(2)
    weights = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="max_cut_edges must be positive"):
        coupling.find_best_penalty_limited_split(
            graph,
            weights,
            lambda cluster: len(cluster),
            max_cut_edges=0,
            coupling_cutoff=1.0,
        )

    with pytest.raises(ValueError, match="coupling_cutoff must be non-negative"):
        coupling.find_best_penalty_limited_split(
            graph,
            weights,
            lambda cluster: len(cluster),
            max_cut_edges=1,
            coupling_cutoff=-1.0,
        )

    with pytest.raises(ValueError, match="cut_penalty_factor must be non-negative"):
        coupling.find_best_penalty_limited_split(
            graph,
            weights,
            lambda cluster: len(cluster),
            max_cut_edges=1,
            coupling_cutoff=1.0,
            cut_penalty_factor=-1.0,
        )
