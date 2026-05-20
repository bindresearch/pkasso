"""Transition matrix utilities for protonation microstates."""

import logging

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .utils import pack_vec

logger = logging.getLogger(__name__)

MISSING = -1000.

###################################################################################
# Microstate transitions/free energy differences from pKa and pH

def calc_charge(pka: float, pH: float = 7.0) -> float:
    """Compute protonated fraction from pKa using Henderson-Hasselbalch."""

    ppos = 1. / ( 1 + 10**(pH-pka) ) # fraction of more positively charged res
    return ppos

def calc_p_up_down(
    pka: float,
    pH: float,
    matrix_def: str,
) -> tuple[float, float]:
    """Compute upward/downward transition values from a pKa at given pH.

    Returns either transition probabilities (`msm`) or free-energy
    differences (`dG`) between protonation states.
    """

    if matrix_def == 'msm':
        p_up = calc_charge(pka,pH=pH) # probability for higher + state
        p_down = 1 - p_up
    elif matrix_def == 'dG':
        p_up = np.log(10) * (pH - pka)
        p_down = -p_up
    else:
        raise
    return p_up, p_down

#############################################################################################
# Transition matrix operations

def calc_state_freqs_sparse(tmatrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """ Compute stationary distribution using power iteration. """
    n_states = tmatrix.shape[0]
    P = csr_matrix(tmatrix)

    pi: NDArray[np.float64] = np.ones(n_states) / n_states
    for idx in range(1000):
        pi = pi @ P
    return pi

def calc_raw_matrix(
    state_strs: list[str],
    state_vecs: list[NDArray[np.int64]],
    ps_all: list[dict[str, dict[int, float]]],
    N_states: int,
    matrix_def: str
    ) -> tuple[list[list[list[float]]], list[list[int]]]:
    """ Assemble raw transition or free-energy entries between microstates. """

    # N_states x N_states (x duplicate predictions)
    matrix_raw: list[list[list[float]]] = [[[] for _ in range(N_states)] for _ in range(N_states)] 

    nonzero_entries = []

    for s_idx, state_vec in enumerate(state_vecs):
        ps_up = ps_all[s_idx]['up']
        ps_down = ps_all[s_idx]['down']

        pss = [ps_up, ps_down]
        dqs = [1, -1]

        # recipes = [
            # [ps_up, 1],
            # [ps_down, -1]
        # ]

        for ps, dq in zip(pss, dqs):
        # for rec in recipes:
        #     ps = rec[0]
        #     dq = rec[1]

            for rel_idx, p in ps.items():
                state_target_vec = state_vec.copy()
                state_target_vec[rel_idx] += dq
                state_target_str = pack_vec(state_target_vec)

                if state_target_str in state_strs:
                    c_target_idx = state_strs.index(state_target_str)

                    if matrix_def == 'msm':
                        matrix_raw[s_idx][c_target_idx].append(p) # from, to; row-stochastic
                        matrix_raw[c_target_idx][s_idx].append(1-p)
                    elif matrix_def == 'dG':
                        matrix_raw[s_idx][c_target_idx].append(p) # from, to
                        matrix_raw[c_target_idx][s_idx].append(-p)
                    else:
                        raise
                    nonzero_entries.append([s_idx,c_target_idx])
                    nonzero_entries.append([c_target_idx,s_idx])
    return matrix_raw, nonzero_entries

def calc_tmatrix(
    state_strs: list[str],
    state_vecs: list[NDArray[np.int64]],
    ps_all: list[dict[str, dict[int, float]]],
    N_states: int,
    ) -> NDArray[np.float64]:
    """ Construct normalized transition matrix between protonation states. """

    tmatrix_raw, nonzero_entries = calc_raw_matrix(state_strs,state_vecs,ps_all,N_states,'msm')

    # N_states x N_states, average over predictions per ij
    tmatrix_mean = np.zeros((N_states,N_states))

    for idx, jdx in nonzero_entries:
        tmatrix_mean[idx,jdx] = np.mean(tmatrix_raw[idx][jdx])

    tmatrix = tmatrix_mean.copy()

    for idx, row in enumerate(tmatrix_mean):
        tmatrix[idx,idx] = np.prod(-1.*row+1) # probability not to transition

    # Normalized tmatrix (probabilities from one state to all other states sums to 1)
    tmatrix_norm = np.zeros((tmatrix.shape))
    for idx, row in enumerate(tmatrix):
        tmatrix_norm[idx] = row / np.sum(row)

    return tmatrix_norm

def calc_dGmatrix(
    state_strs: list[str],
    state_vecs: list[NDArray[np.int64]],
    ps_all: list[dict[str, dict[int, float]]],
    N_states: int,
    ) -> NDArray[np.float64]:
    """Construct matrix of pairwise free-energy differences between states."""

    matrix_raw, nonzero_entries = calc_raw_matrix(state_strs,state_vecs,ps_all,N_states,'dG')

    # N_states x N_states, average over predictions per ij
    matrix_mean = np.zeros((N_states,N_states)) - 1000 

    for idx, jdx in nonzero_entries:
        matrix_mean[idx,jdx] = np.mean(matrix_raw[idx][jdx])
    return matrix_mean

def find_dGclusters(dG_matrix: NDArray[np.float64]) -> list[list[int]]:
    """
    Identify all clusters (connected components) in dG_matrix.

    Parameters
    ----------
    dG_matrix : (N, N) ndarray
        dG_matrix[i, j] ≈ G_j - G_i, or MISSING if unavailable

    Returns
    -------
    clusters : list[list[int]]
        Each sublist contains the indices of states in that connected cluster
    """
    dG_matrix = np.asarray(dG_matrix)
    N = dG_matrix.shape[0]

    # Build undirected graph
    Gr = nx.Graph()
    Gr.add_nodes_from(range(N))

    for i in range(N):
        for j in range(i+1, N):
            if dG_matrix[i, j] != MISSING or dG_matrix[j, i] != MISSING:
                Gr.add_edge(i, j)

    # Extract connected components
    clusters = [list(c) for c in nx.connected_components(Gr)]
    return clusters

def remove_orphans(
    dG_clusters: list[list[int]],
    state_strs: list[str],
    dG_matrix: NDArray[np.float64],
    ) -> tuple[list[str], NDArray[np.float64]]:
    """Keep the largest connected cluster and discard isolated states."""

    nmax = 0
    keep_idx = 0
    for idx, dG_cluster in enumerate(dG_clusters):
        if len(dG_cluster) > nmax:
            keep_idx = idx
            nmax = len(dG_cluster)
    keep_ids = np.array(dG_clusters[keep_idx])

    state_strs_keep = [state_strs[idx] for idx in keep_ids]
    dG_matrix_keep = dG_matrix[np.ix_(keep_ids, keep_ids)]
    return state_strs_keep, dG_matrix_keep

def check_connectivity(dG_matrix: NDArray[np.float64]) -> bool:
    """Check whether the free-energy difference graph is fully connected."""

    N = dG_matrix.shape[0]
    G = nx.Graph()

    if N == 1:
        return True

    for i in range(N):
        for j in range(i+1, N):
            if dG_matrix[i, j] != MISSING:# and i != j: # 
                G.add_edge(i, j)

    connected = bool(nx.is_connected(G))
    return connected

def calc_populations(Gs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Boltzmann populations from free energies."""
    Z: NDArray[np.float64] = np.sum(np.exp(-Gs))
    pops: NDArray[np.float64] = np.exp(-Gs) / Z # Boltzmann weights
    return pops

def calc_freqs_from_states(
    state_strs: list[str],
    state_vecs: list[NDArray[np.int64]],
    ps_all: list[dict[str, dict[int, float]]],
    matrix_def: str,
    ) -> tuple[list[str], NDArray[np.float64]]:
    """Compute microstate frequencies using MSM or dG reconstruction."""

    N_states = len(state_vecs)
    if matrix_def == 'msm':
        tmatrix = calc_tmatrix(state_strs,state_vecs,ps_all,N_states)
        state_freqs = calc_state_freqs_sparse(tmatrix)
    elif matrix_def == 'dG':
        dGmatrix = calc_dGmatrix(state_strs,state_vecs,ps_all,N_states)
        dG_clusters = find_dGclusters(dGmatrix)
        state_strs, dGmatrix = remove_orphans(dG_clusters, state_strs, dGmatrix)
        is_connected = check_connectivity(dGmatrix)
        if not is_connected:
            raise ValueError('Matrix not connected (or not symmetric)')
        Gs = reconstruct_free_energies_weighted(dGmatrix)
        state_freqs = calc_populations(Gs)
    else:
        raise ValueError
    return state_strs, state_freqs

def calc_state_diffs(
    state_strs: list[str],
    state_vecs: list[NDArray[np.int64]],
    indices: list[int],
    base_lib: dict[str, dict[int, float]],
    acid_lib: dict[str, dict[int, float]],
    # mols_lib: dict[str, Mol],
    pH: float = 7.0,
    matrix_def: str = 'dG',
    ) -> list[dict[str, dict[int, float]]]:
    """ Compute state transition values from predicted acid/base pKa values. """

    ps_all = [] # pH specific

    for state_str, state_vec in zip(state_strs, state_vecs):
        logger.debug('='*20)
        logger.debug(f'{state_str}')
        ps_up = {}
        ps_down = {}

        base = base_lib[state_str]
        acid = acid_lib[state_str]

        for map_idx, pka in base.items():
            if map_idx not in indices: # Excluded at the start
                continue
            p_up, p_down = calc_p_up_down(pka,pH,matrix_def)#,q+1,q)

            rel_idx = indices.index(map_idx)
            logger.debug(f'rel_idx:{rel_idx} | map_idx:{map_idx} | base {pka} up:{p_up:.2f} stay:{p_down:.2f}')
            if state_vec[rel_idx] <= 1:
                ps_up[rel_idx] = p_up

        for map_idx, pka in acid.items():
            if map_idx not in indices: # Excluded at the start
                continue
            p_up, p_down = calc_p_up_down(pka,pH,matrix_def)#,q-1,q)

            rel_idx = indices.index(map_idx)
            logger.debug(f'rel_idx:{rel_idx} | map_idx:{map_idx} | acid {pka} stay:{p_up:.2f} down:{p_down:.2f}')
            if state_vec[rel_idx] >= 1:
                ps_down[rel_idx] = p_down

        ps = {
            'up' : ps_up,
            'down' : ps_down,
        }
        ps_all.append(ps)

    return ps_all

####

def reconstruct_free_energies_weighted(
    dG_matrix: NDArray[np.float64],
    sigma0: float = 0.2,
    alpha: float = 0.1,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """
    Reconstruct absolute free energies from incomplete pairwise deltaG data
    using iteratively reweighted least squares.

    Weighting scheme:
        sigma_ij = sigma0 + alpha * mean(G_i, G_j)

        weight_ij = 1 / sigma_ij^2

    After each iteration the free energies are shifted such that:
        min(G) = 0

    This makes the weighting gauge-invariant and ensures:
        sigma_ij > 0

    Notes
    -----
    - Larger free energies receive lower weights.
    - Lower free energies receive higher weights.
    """

    N = dG_matrix.shape[0]

    if N == 1:
        return np.array([0.0])

    # --------------------------------------------------------------
    # Build pair list
    # --------------------------------------------------------------
    pairs = []

    for i in range(N):
        for j in range(i + 1, N):

            val = dG_matrix[i, j]

            if np.isnan(val) or val == MISSING:
                continue

            pairs.append((i, j, val))

    if not pairs:
        raise ValueError("No valid transitions found.")

    # --------------------------------------------------------------
    # Initial ordinary least-squares solution
    # --------------------------------------------------------------
    rows = []
    rhs = []

    for i, j, val in pairs:

        row = np.zeros(N)
        row[i] = -1.0
        row[j] = 1.0

        rows.append(row)
        rhs.append(val)

    A = np.vstack(rows)
    b = np.array(rhs)

    # Temporary gauge fixing for solve
    A_reduced = A[:, 1:]

    G_reduced, *_ = np.linalg.lstsq(A_reduced, b, rcond=None)

    G = np.zeros(N)
    G[1:] = G_reduced

    # Shift so min(G) = 0
    G -= np.min(G)

    # --------------------------------------------------------------
    # Iterative weighted least squares
    # --------------------------------------------------------------
    for _ in range(max_iter):

        weighted_rows = []
        weighted_rhs = []

        for i, j, val in pairs:

            # Mean free energy
            scale = 0.5 * (G[i] + G[j])

            sigma_ij = sigma0 + alpha * scale

            weight = 1.0 / (sigma_ij ** 2)

            row = np.zeros(N)
            row[i] = -1.0
            row[j] = 1.0

            weighted_rows.append(np.sqrt(weight) * row)
            weighted_rhs.append(np.sqrt(weight) * val)

        A_w = np.vstack(weighted_rows)
        b_w = np.array(weighted_rhs)

        A_w_reduced = A_w[:, 1:]

        G_reduced_new, *_ = np.linalg.lstsq(
            A_w_reduced,
            b_w,
            rcond=None
        )

        G_new = np.zeros(N)
        G_new[1:] = G_reduced_new

        # Shift gauge so minimum free energy is zero
        G_new -= np.min(G_new)

        # Convergence check
        if np.linalg.norm(G_new - G) < tol:
            G = G_new
            break

        G = G_new

    return G