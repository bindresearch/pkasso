"""Helper functions to test pKa coupling between protonation sites."""

import logging
import itertools
import math
from collections.abc import Callable
from typing import cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def construct_state_vectors_single(indices: list[int], q_options: NDArray[np.int64]) -> list[NDArray[np.int64]]:
    """
    Construct single-site perturbation state vectors.

    Generates protonation state vectors used for coupling analysis.
    The neutral reference state (all sites neutral) is included, along
    with states where exactly one site is protonated or deprotonated,
    provided that transition is allowed by `q_options`.

    Parameters
    ----------
    indices
        Atom map indices for protonable sites.
    q_options
        Array indicating which sites can be protonated or deprotonated.

    Returns
    -------
    state_vecs
        List of protonation state vectors.
    """

    state_vecs = []
    state_vec = np.ones((len(indices)), dtype=int)
    state_vecs.append(state_vec)  # Add neutral state

    for rel_idx, map_idx in enumerate(indices):
        for q in [0, 2]:
            if q_options[rel_idx][q] == 1:
                state_vec = np.ones((len(indices)), dtype=int)
                state_vec[rel_idx] = q
                state_vecs.append(state_vec)
    return state_vecs


def compare_pkas(
    indices: list[int],
    q_options: NDArray[np.int64],
    state_str0: str,
    state_str1: str,
    base_lib: dict[str, dict[int, float]],
    acid_lib: dict[str, dict[int, float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute pKa differences between two protonation states.

    For each site, this function compares predicted acid and base pKa
    values between a reference state (`state_str0`) and a perturbed
    state (`state_str1`). Missing predictions are treated as large
    differences to indicate strong coupling.

    Parameters
    ----------
    indices
        Atom map indices for protonable sites.
    q_options
        Array indicating which sites can be protonated or deprotonated.
    state_str0
        State string of reference state.
    state_str1
        State string of perturbed state.
    base_lib:
        Library of molgpka pKa base predictions (specific to indices).
    acid_lib:
        Library of molgpka pKa acid predictions (specific to indices).

    Returns
    -------
    base_pka_diff : NDArray[np.float64]
        Absolute differences in predicted base pKa values per site.
    acid_pka_diff : NDArray[np.float64]
        Absolute differences in predicted acid pKa values per site.
    """

    base_pka_diff = np.zeros(len(indices))
    acid_pka_diff = np.zeros(len(indices))
    for rel_idx, map_idx in enumerate(indices):
        if q_options[rel_idx][2] == 1:  # allowed for base
            if (map_idx in base_lib[state_str0]) and (map_idx in base_lib[state_str1]):
                base_pka_diff[rel_idx] = abs(base_lib[state_str1][map_idx] - base_lib[state_str0][map_idx])
            else:
                base_pka_diff[rel_idx] = 10.0  # one disappeared
        if q_options[rel_idx][0] == 1:  # allowed for acid
            if (map_idx in acid_lib[state_str0]) and (map_idx in acid_lib[state_str1]):
                acid_pka_diff[rel_idx] = abs(acid_lib[state_str1][map_idx] - acid_lib[state_str0][map_idx])
            else:
                acid_pka_diff[rel_idx] = 10.0  # one disappeared
    return base_pka_diff, acid_pka_diff


def construct_coupling_matrix(
    indices: list[int],
    state_strs: list[str],
    state_vecs: list[NDArray[np.int64]],
    base_pka_diffs: dict[str, NDArray[np.float64]],
    acid_pka_diffs: dict[str, NDArray[np.float64]],
    coupling_cutoff: float,
) -> NDArray[np.int64]:
    """
    Build a site-site coupling matrix from pKa perturbations.

    The matrix records how strongly protonating or deprotonating one
    site affects the predicted pKa values of other sites. Entries are
    incremented when pKa differences exceed the specified cutoff.

    Parameters
    ----------
    indices
        Atom map indices for protonable sites.
    state_strs
        State strings for microstates.
    state_vecs
        State vectors corresponding to state strings.
    base_pka_diffs
        Library of base pKa differences w.r.t. the neutral reference state.
        Each entry of the dictionary is [state_str, pKa_diffs].
    acid_pka_diffs
        Library of acid pKa differences w.r.t. the neutral reference state.
        Each entry of the dictionary is [state_str, pKa_diffs].
    coupling_cutoff
        pKa difference cutoff above which two sites are coupled.

    Returns
    -------
    coupling_matrix
        Square matrix indicating pairwise coupling strength between sites.
    """

    coupling_matrix: NDArray[np.int64] = np.zeros((len(indices), len(indices)), dtype=np.int64)

    for state_str, state_vec in zip(state_strs[1:], state_vecs[1:]):
        changed_rel_idx = np.where(state_vec != 1)[0][0]
        coupling_matrix[changed_rel_idx] += np.where(base_pka_diffs[state_str] >= coupling_cutoff, 1, 0)
        coupling_matrix[changed_rel_idx] += np.where(acid_pka_diffs[state_str] >= coupling_cutoff, 1, 0)
    return coupling_matrix


def construct_coupling_weight_matrix(
    indices: list[int],
    state_strs: list[str],
    state_vecs: list[NDArray[np.int64]],
    base_pka_diffs: dict[str, NDArray[np.float64]],
    acid_pka_diffs: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Build a site-site coupling weight matrix from raw pKa perturbations.

    Each directed entry stores the largest acid/base pKa change observed for a
    target site when another site is perturbed. Multiple perturbation states of
    the same site are combined by taking the maximum observed response.
    """

    coupling_weights: NDArray[np.float64] = np.zeros((len(indices), len(indices)), dtype=np.float64)

    for state_str, state_vec in zip(state_strs[1:], state_vecs[1:]):
        changed_rel_idx = np.where(state_vec != 1)[0][0]
        pka_diffs = np.maximum(base_pka_diffs[state_str], acid_pka_diffs[state_str])
        coupling_weights[changed_rel_idx] = np.maximum(coupling_weights[changed_rel_idx], pka_diffs)

    return coupling_weights


def threshold_coupling_weights(M: NDArray[np.float64], coupling_cutoff: float) -> NDArray[np.int64]:
    """Convert raw pKa coupling weights into a thresholded coupling matrix."""

    if coupling_cutoff < 0:
        raise ValueError("coupling_cutoff must be non-negative.")
    return np.asarray(M >= coupling_cutoff, dtype=np.int64)


def validate_coupling_matrix(M: NDArray[np.float64] | NDArray[np.int64]) -> NDArray[np.float64] | NDArray[np.int64]:
    """Validate and return a square coupling matrix."""

    matrix = cast(NDArray[np.float64] | NDArray[np.int64], np.asarray(M))
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Coupling matrix must be square.")
    return matrix


def coupling_matrix_to_graph(M: NDArray[np.float64] | NDArray[np.int64]) -> nx.Graph:
    """
    Convert a directed coupling matrix into an undirected site graph.

    Any nonzero sensitivity in either direction is treated as one site-site
    connection. This preserves the previous connected-component semantics while
    allowing NetworkX to perform more nuanced graph partitioning.
    """

    validate_coupling_matrix(M)
    n = M.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] != 0 or M[j, i] != 0:
                graph.add_edge(i, j)

    return graph


def coupling_weights_to_graph(
    M: NDArray[np.float64],
    coupling_cutoff: float,
    nodes: list[int] | None = None,
) -> nx.Graph:
    """
    Build an undirected weighted graph from pKa coupling weights.

    Edges are included when the maximum pKa response in either direction meets
    ``coupling_cutoff``. The edge ``weight`` stores that undirected maximum.
    """

    if coupling_cutoff < 0:
        raise ValueError("coupling_cutoff must be non-negative.")

    validate_coupling_matrix(M)
    if nodes is None:
        nodes = list(range(M.shape[0]))

    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    for idx, i in enumerate(nodes):
        for j in nodes[idx + 1 :]:
            weight = float(max(M[i, j], M[j, i]))
            if weight >= coupling_cutoff:
                graph.add_edge(i, j, weight=weight)

    return graph


def cluster_coupling_matrix(M: NDArray[np.int64]) -> list[list[int]]:
    """
    Partition sites into clusters based on coupling connectivity.

    Identify connected components in the coupling matrix, grouping sites that
    influence each other into clusters.

    Parameters
    ----------
    coupling_matrix
        Square matrix indicating pairwise coupling strength between sites.

    Returns
    -------
    clusters
        Lists of site indices belonging to each coupling cluster.
    """

    graph = coupling_matrix_to_graph(M)
    clusters = [sorted(c) for c in nx.connected_components(graph)]
    return sorted(clusters, key=lambda c: c[0])


def cutset_penalty(M: NDArray[np.float64], cutset: tuple[tuple[int, int], ...]) -> float:
    """Sum undirected pKa penalties for severing a set of graph edges."""

    validate_coupling_matrix(M)
    return float(sum(max(M[i, j], M[j, i]) for i, j in cutset))


def find_best_penalty_limited_split(
    graph: nx.Graph,
    coupling_weights: NDArray[np.float64],
    cluster_state_count: Callable[[list[int]], int],
    max_cut_edges: int,
    coupling_cutoff: float,
    cut_penalty_factor: float = 1.7,
) -> list[list[int]] | None:
    """
    Find the best acceptable split by enumerating small edge cutsets.

    Candidate cutsets up to ``max_cut_edges`` are tested with NetworkX by
    removing the edges and checking whether the graph disconnects. A cutset is
    acceptable when its pKa penalty is no larger than
    ``cut_penalty_factor * coupling_cutoff * sqrt(len(cutset))``. Among
    acceptable candidates, the split minimizing the summed child-cluster state
    count is selected.
    """

    if max_cut_edges < 1:
        raise ValueError("max_cut_edges must be positive.")
    if coupling_cutoff < 0:
        raise ValueError("coupling_cutoff must be non-negative.")
    if cut_penalty_factor < 0:
        raise ValueError("cut_penalty_factor must be non-negative.")

    edges = [tuple(sorted(edge)) for edge in graph.edges()]
    best_components = None
    best_score = None

    for n_cut_edges in range(1, max_cut_edges + 1):
        for cutset_raw in itertools.combinations(edges, n_cut_edges):
            cutset = tuple(cutset_raw)
            penalty = cutset_penalty(coupling_weights, cutset)
            max_cut_penalty = cut_penalty_factor * coupling_cutoff * math.sqrt(n_cut_edges)
            if penalty > max_cut_penalty:
                continue

            cut_graph = graph.copy()
            cut_graph.remove_edges_from(cutset)
            components = [sorted(component) for component in nx.connected_components(cut_graph)]
            if len(components) < 2:
                continue

            components = sorted(components, key=lambda c: c[0])
            state_count = sum(cluster_state_count(component) for component in components)
            score = (state_count, penalty, n_cut_edges)
            if best_score is None or score < best_score:
                best_score = score
                best_components = components

    return best_components
