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

def calc_dGmatrix(state_vecs,state_strs,ps_all,N_states):
    """ Matrix of free energy differences between protonation states """

    matrix_raw = [[[] for _ in range(N_states)] for _ in range(N_states)] # N_states x N_states (x duplicate predictions)

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

                        matrix_raw[s_idx][c_target_idx].append(p) # from, to
                        matrix_raw[c_target_idx][s_idx].append(-p)
                        nonzero_entries.append([s_idx,c_target_idx])
                        nonzero_entries.append([c_target_idx,s_idx])

    matrix_mean = np.zeros((N_states,N_states)) - 1000 # N_states x N_states, average over predictions per ij

    for idx, jdx in nonzero_entries:
        matrix_mean[idx,jdx] = np.mean(matrix_raw[idx][jdx])
    return matrix_mean

def calc_Fs(matrix,max_visited=5):
    N_states = matrix.shape[0]
    # print(f'N_states: {N_states}')
    visited_counter = np.zeros((N_states))
    # Fs = np.zeros(N_states)
    Fs = [[] for _ in range(N_states)]
    # row = matrix[0]
    # ids_targets = np.where(row!=-1000)[0]
    # pops[ids_targets] = row[ids_targets]
    # visited.append(0)
    ids_origins = [0]
    Fs[0].append(0.)
    while len(ids_origins) > 0:
        ids_new_origins = np.array([],dtype=int)
        for idx in ids_origins:
            idx = int(idx)
            # print(idx)
            # print(Fs[idx])
            # visited.append(idx)
            visited_counter[idx] += 1
            row = matrix[idx]
            # print(row)
            ids_targets = np.where(row!=-1000)[0]
            # print(f'ids_targets: {ids_targets}')
            for idx_target in ids_targets:
                # print(np.mean(Fs[idx]))
                Fs[idx_target].append(np.mean(Fs[idx]) + row[idx_target])
                # print(idx_target, Fs[idx_target])
            # Fs[ids_targets] = Fs[idx] + row[ids_targets]
            # print(Fs[ids_targets])
            ids_new_origins = np.append(ids_new_origins,ids_targets)
        ids_origins = []
        for idx in np.unique(ids_new_origins):
            # if idx not in visited:
            if visited_counter[idx] < max_visited:# not in visited:
                ids_origins.append(idx)
        ids_origins = np.array(ids_origins,dtype=int)
        # print(f'New origins: {ids_origins}')
    # print(f'Visited: {visited_counter}')
    # print(f'Fs: {Fs}')
    Fs_m = np.array([np.mean(F) for F in Fs])
    Fs_stds = np.array([np.std(F) for F in Fs])
    # print(f'Fs_means: {Fs_m}')
    # print(f'Fs_stds: {Fs_stds}')
    Fs_m -= np.min(Fs_m)
    # print(Fs)
    # print(Fs)
    return Fs_m

def calc_populations(Fs):
    Z = np.sum(np.exp(-Fs))
    pops = np.exp(-Fs) / Z # Boltzmann weights
    # print(f'Boltzmann weights: {pops}')
    return pops

    # matrix = matrix_mean.copy()

    # for idx, row in enumerate(matrix_mean):
        # tmatrix[idx,idx] = np.prod(-1.*row+1) # probability not to transition

    # Normalized tmatrix (probabilities from one state to all other states sums to 1)
    # tmatrix_norm = []
    # for row in tmatrix:
        # tmatrix_norm.append(row / np.sum(row))
    
    # tmatrix_norm = np.array(tmatrix_norm)

# def calc_tmatrix(state_vecs,state_strs,ps_all,N_states):
#     """ Transition matrix between molecule protonation states"""

#     tmatrix_raw = [[[] for _ in range(N_states)] for _ in range(N_states)] # N_states x N_states (x duplicate predictions)

#     nonzero_entries = []

#     for s_idx, state_vec in enumerate(state_vecs):
#         ps_up = ps_all[s_idx,0]
#         ps_down = ps_all[s_idx,1]

#         recipes = [
#             [ps_up, 1],
#             [ps_down, -1]
#         ]

#         for rec in recipes:
#             ps = rec[0]
#             dq = rec[1]

#             for at_idx, p in enumerate(ps):
#                 if p > -1.:
#                     p = float(p)
#                     state_target_vec = state_vec.copy()
#                     state_target_vec[at_idx] += dq
#                     state_target_str = pack_vec(state_target_vec)

#                     if state_target_str in state_strs:
#                         c_target_idx = state_strs.index(state_target_str)

#                         tmatrix_raw[s_idx][c_target_idx].append(p) # from, to; row-stochastic
#                         tmatrix_raw[c_target_idx][s_idx].append(1-p)
#                         nonzero_entries.append([s_idx,c_target_idx])
#                         nonzero_entries.append([c_target_idx,s_idx])


#     tmatrix_mean = np.zeros((N_states,N_states)) # N_states x N_states, average over predictions per ij

#     for idx, jdx in nonzero_entries:
#         tmatrix_mean[idx,jdx] = np.mean(tmatrix_raw[idx][jdx])

#     tmatrix = tmatrix_mean.copy()

#     for idx, row in enumerate(tmatrix_mean):
#         tmatrix[idx,idx] = np.prod(-1.*row+1) # probability not to transition

#     # Normalized tmatrix (probabilities from one state to all other states sums to 1)
#     tmatrix_norm = []
#     for row in tmatrix:
#         tmatrix_norm.append(row / np.sum(row))
    
#     tmatrix_norm = np.array(tmatrix_norm)

#     return tmatrix_norm