"""Helper functions to test pKa coupling between protonation sites."""

import logging

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
    state_vec = np.ones((len(indices)),dtype=int)
    state_vecs.append(state_vec) # Add neutral state

    for rel_idx, map_idx in enumerate(indices):
        for q in [0,2]:
            if q_options[rel_idx][q] == 1:
                state_vec = np.ones((len(indices)),dtype=int)
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
        if (q_options[rel_idx][2] == 1): # allowed for base
            if (map_idx in base_lib[state_str0]) and (map_idx in base_lib[state_str1]):
                base_pka_diff[rel_idx] = abs(base_lib[state_str1][map_idx] - base_lib[state_str0][map_idx])
            else:
                base_pka_diff[rel_idx] = 10. # one disappeared
        if (q_options[rel_idx][0] == 1): # allowed for acid
            if (map_idx in acid_lib[state_str0]) and (map_idx in acid_lib[state_str1]):
                acid_pka_diff[rel_idx] = abs(acid_lib[state_str1][map_idx] - acid_lib[state_str0][map_idx])
            else:
                acid_pka_diff[rel_idx] = 10. # one disappeared
    return base_pka_diff, acid_pka_diff

def construct_coupling_matrix(
    indices: list[int],
    state_strs: list[str],
    state_vecs: list[NDArray[np.int64]],
    base_pka_diffs: dict[str, NDArray[np.float64]],
    acid_pka_diffs: dict[str, NDArray[np.float64]],
    coupling_cutoff: float
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

    coupling_matrix: NDArray[np.int64] = np.zeros((len(indices),len(indices)), dtype=np.int64)

    for state_str, state_vec in zip(state_strs[1:], state_vecs[1:]):
        changed_rel_idx = np.where(state_vec != 1)[0][0]
        coupling_matrix[changed_rel_idx] += np.where(base_pka_diffs[state_str] >= coupling_cutoff, 1, 0)
        coupling_matrix[changed_rel_idx] += np.where(acid_pka_diffs[state_str] >= coupling_cutoff, 1, 0)
    return coupling_matrix

def cluster_coupling_matrix(M: NDArray[np.int64]) -> list[list[int]]:
    """
    Partition sites into clusters based on coupling connectivity.

    Identify connected components in the coupling matrix, grouping
    sites that influence each other into clusters.

    Parameters
    ----------
    coupling_matrix
        Square matrix indicating pairwise coupling strength between sites.

    Returns
    -------
    clusters
        Lists of site indices belonging to each coupling cluster.
    """

    n = M.shape[0]
    visited = set()
    clusters = []

    def dfs(i: int, cluster: set[int]) -> None:
        for j in range(n):
            if j not in visited and (M[i, j] != 0 or M[j, i] != 0):
                visited.add(j)
                cluster.add(j)
                dfs(j, cluster)

    for i in range(n):
        if i not in visited:
            visited.add(i)
            cluster = {i}
            dfs(i, cluster)
            clusters.append(cluster)
    clusters_out = [list(c) for c in clusters]
    return clusters_out