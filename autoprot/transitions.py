import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
from .utils import pack_vec

MISSING = -1000.

###################################################################################
# Microstate transitions/free energy differences from pKa and pH

def calc_charge(pka,pH=7.):
    """ Hendersson-Hasselbalch eq. """

    ppos = 1. / ( 1 + 10**(pH-pka) ) # fraction of more positively charged res
    return ppos

def calc_p_up_down(pka,pH,matrix_def):
    """ Calc either transition probability or free energy difference between microstates """

    if matrix_def == 'msm':
        p_up = calc_charge(pka,pH=pH) # probability for higher + state
        p_down = 1 - p_up
    elif matrix_def == 'dG':
        p_up = np.log(10) * (pH - pka) # -ln(p+/p0)
        p_down = np.log(10) * (pka - pH) # -ln(p0/p+)
    else:
        raise
    return p_up, p_down

def calc_state_diffs(state_strs, state_vecs, base_lib, acid_lib, indices, pH=7.,matrix_def='dG',
                verbose=False):
    """ Calc state differences (free energy or MSM) from acid/base pka values for given pH """

    ps_all = [] # pH specific

    for state_str, state_vec in zip(state_strs, state_vecs):
        if verbose:
            print('='*20)
            print(f'{state_str}')
        ps_up = {}
        ps_down = {}

        base = base_lib[state_str]
        acid = acid_lib[state_str]
        for map_idx, pka in base.items():
            if map_idx not in indices: # Excluded at the start
                continue
            p_up, p_down = calc_p_up_down(pka,pH,matrix_def)

            rel_idx = indices.index(map_idx)
            if verbose:
                print(f'rel_idx:{rel_idx} | map_idx:{map_idx} | base {pka} up:{p_up:.2f} stay:{p_down:.2f}')
            if state_vec[rel_idx] <= 1:
                ps_up[rel_idx] = p_up

        for map_idx, pka in acid.items():
            if map_idx not in indices: # Excluded at the start
                continue
            p_up, p_down = calc_p_up_down(pka,pH,matrix_def)

            rel_idx = indices.index(map_idx)
            if verbose:
                print(f'rel_idx:{rel_idx} | map_idx:{map_idx} | acid {pka} stay:{p_up:.2f} down:{p_down:.2f}')
            if state_vec[rel_idx] >= 1:
                ps_down[rel_idx] = p_down

        ps = {
            'up' : ps_up,
            'down' : ps_down,
        }
        ps_all.append(ps)

    return ps_all

#############################################################################################
# Transition matrix operations

def calc_state_freqs(tmatrix):
    w, v = np.linalg.eig(tmatrix.T)
    idx = np.argmin(np.abs(w - 1))
    pi = np.real(v[:, idx])
    pi = pi / pi.sum()
    return pi

def calc_state_freqs_sparse(tmatrix):
    P = csr_matrix(tmatrix)

    pi = np.ones(P.shape[0]) / P.shape[0]
    for idx in range(1000):
        pi = pi @ P
    return pi

def calc_state_freqs_power_iter(tmatrix):
    pi = np.ones(tmatrix.shape[0]) / tmatrix.shape[0]

    for _ in range(10_000):
        pi = pi @ tmatrix

    pi /= pi.sum()
    return pi

def calc_raw_matrix(state_strs,state_vecs,ps_all,N_states,matrix_def):
    """ Make raw matrix between microstates from ps_up and ps_down """

    # N_states x N_states (x duplicate predictions)
    matrix_raw = [[[] for _ in range(N_states)] for _ in range(N_states)] 

    nonzero_entries = []

    for s_idx, state_vec in enumerate(state_vecs):
        ps_up = ps_all[s_idx]['up']
        ps_down = ps_all[s_idx]['down']

        recipes = [
            [ps_up, 1],
            [ps_down, -1]
        ]

        for rec in recipes:
            ps = rec[0]
            dq = rec[1]

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

def calc_tmatrix(state_strs,state_vecs,ps_all,N_states):
    """ Transition matrix between molecule protonation states"""

    tmatrix_raw, nonzero_entries = calc_raw_matrix(state_strs,state_vecs,ps_all,N_states,'msm')

    # N_states x N_states, average over predictions per ij
    tmatrix_mean = np.zeros((N_states,N_states))

    for idx, jdx in nonzero_entries:
        tmatrix_mean[idx,jdx] = np.mean(tmatrix_raw[idx][jdx])

    tmatrix = tmatrix_mean.copy()

    for idx, row in enumerate(tmatrix_mean):
        tmatrix[idx,idx] = np.prod(-1.*row+1) # probability not to transition

    # Normalized tmatrix (probabilities from one state to all other states sums to 1)
    tmatrix_norm = []
    for row in tmatrix:
        tmatrix_norm.append(row / np.sum(row))
    
    tmatrix_norm = np.array(tmatrix_norm)

    return tmatrix_norm

def calc_dGmatrix(state_strs,state_vecs,ps_all,N_states):
    """ Matrix of free energy differences between protonation states """

    matrix_raw, nonzero_entries = calc_raw_matrix(state_strs,state_vecs,ps_all,N_states,'dG')

    # N_states x N_states, average over predictions per ij
    matrix_mean = np.zeros((N_states,N_states)) - 1000 

    for idx, jdx in nonzero_entries:
        matrix_mean[idx,jdx] = np.mean(matrix_raw[idx][jdx])
    return matrix_mean

def find_dGclusters(dG_matrix):
    """
    Identify all clusters (connected components) in dG_matrix.

    Parameters
    ----------
    dG_matrix : (N, N) ndarray
        dG_matrix[i, j] ≈ G_j - G_i, or MISSING if unavailable

    Returns
    -------
    clusters : list of lists
        Each sublist contains the indices of states in that connected cluster
    """
    dG_matrix = np.asarray(dG_matrix)
    N = dG_matrix.shape[0]

    # Build undirected graph
    Gr = nx.Graph()
    Gr.add_nodes_from(range(N))

    for i in range(N):
        for j in range(i+1, N):
            if dG_matrix[i, j] != MISSING:
                Gr.add_edge(i, j)

    # Extract connected components
    clusters = [list(c) for c in nx.connected_components(Gr)]
    return clusters

def remove_orphans(dG_clusters, state_strs, dG_matrix):
    """ Remove disconnected states (never visited) """

    nmax = 0
    for idx, dG_cluster in enumerate(dG_clusters):
        if len(dG_cluster) > nmax:
            keep_idx = idx
            nmax = len(dG_cluster)
    keep_ids = np.array(dG_clusters[keep_idx])

    state_strs_keep = [state_strs[idx] for idx in keep_ids]
    dG_matrix_keep = dG_matrix[np.ix_(keep_ids, keep_ids)]
    return state_strs_keep, dG_matrix_keep

def check_connectivity(dG_matrix):
    """ Double check that matrix is fully connected """

    N = dG_matrix.shape[0]
    G = nx.Graph()

    if N == 1:
        return True

    for i in range(N):
        for j in range(N):
            if dG_matrix[i, j] != MISSING and i != j:
                G.add_edge(i, j)

    return nx.is_connected(G)

def reconstruct_free_energies_incomplete_half(dG_matrix):
    N = dG_matrix.shape[0]
    rows, rhs = [], []

    if N == 1:
        return np.array([0.])

    for i in range(N):
        for j in range(i+1, N):
            if dG_matrix[i, j] == MISSING:
                continue

            row = np.zeros(N)
            row[i] = -1.0
            row[j] =  1.0
            rows.append(row)
            rhs.append(dG_matrix[i, j])

    if not rows:
        raise ValueError("No valid transitions found.")

    A = np.vstack(rows)
    b = np.array(rhs)

    A = A[:, 1:]
    G_reduced, *_ = np.linalg.lstsq(A, b, rcond=None)

    G = np.zeros(N)
    G[1:] = G_reduced
    return G

def calc_populations(Gs):
    Z = np.sum(np.exp(-Gs))
    pops = np.exp(-Gs) / Z # Boltzmann weights
    return pops

def calc_freqs_from_states(state_strs,state_vecs,ps_all,matrix_def):
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
            raise ValueError('Matrix not connected')
        Gs = reconstruct_free_energies_incomplete_half(dGmatrix)
        state_freqs = calc_populations(Gs)
    return state_strs, state_freqs
