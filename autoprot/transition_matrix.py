import numpy as np
from scipy.sparse import csr_matrix

from .utils import pack_vec

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

def calc_tmatrix(state_vecs,state_strs,ps_all,N_states):
    """ Transition matrix between molecule protonation states"""

    tmatrix_raw = [[[] for _ in range(N_states)] for _ in range(N_states)] # N_states x N_states (x duplicate predictions)

    nonzero_entries = []

    for s_idx, state_vec in enumerate(state_vecs):
        ps_up = ps_all[s_idx,0]
        ps_down = ps_all[s_idx,1]

        recipes = [
            [ps_up, 1],
            [ps_down, -1]
        ]

        for rec in recipes:
            ps = rec[0]
            dq = rec[1]

            for at_idx, p in enumerate(ps):
                if p > -1.:
                    p = float(p)
                    state_target_vec = state_vec.copy()
                    state_target_vec[at_idx] += dq
                    state_target_str = pack_vec(state_target_vec)

                    if state_target_str in state_strs:
                        c_target_idx = state_strs.index(state_target_str)

                        tmatrix_raw[s_idx][c_target_idx].append(p) # from, to; row-stochastic
                        tmatrix_raw[c_target_idx][s_idx].append(1-p)
                        nonzero_entries.append([s_idx,c_target_idx])
                        nonzero_entries.append([c_target_idx,s_idx])


    tmatrix_mean = np.zeros((N_states,N_states)) # N_states x N_states, average over predictions per ij

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