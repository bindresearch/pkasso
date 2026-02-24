import numpy as np

def construct_state_vectors_single(indices, q_options, verbose=False):
    state_vecs = []
    state_vec = np.ones((len(indices)),dtype=int) #[1 for _ in indices]
    state_vecs.append(state_vec)

    for rel_idx, map_idx in enumerate(indices):
        for q in [0,2]:
            if q_options[q][rel_idx] == 1:
                state_vec = np.ones((len(indices)),dtype=int) #[1 for _ in indices]
                state_vec[rel_idx] = q
                state_vecs.append(state_vec)
    return state_vecs

def compare_pkas(indices, q_options, state_str0, state_str1, base_lib, acid_lib):
    base_pka_diff = np.zeros((len(indices)))
    acid_pka_diff = np.zeros((len(indices)))
    for rel_idx, at_idx in enumerate(indices):
        if (q_options[2][rel_idx] == 1): # allowed for base
            if (at_idx in base_lib[state_str0]) and (at_idx in base_lib[state_str1]):
                base_pka_diff[rel_idx] = abs(base_lib[state_str1][at_idx] - base_lib[state_str0][at_idx])
            else:
                base_pka_diff[rel_idx] = 10. # one disappeared
        if (q_options[0][rel_idx] == 1): # allowed for acid
            if (at_idx in acid_lib[state_str0]) and (at_idx in acid_lib[state_str1]):
                acid_pka_diff[rel_idx] = abs(acid_lib[state_str1][at_idx] - acid_lib[state_str0][at_idx])
            else:
                acid_pka_diff[rel_idx] = 10. # one disappeared
    return base_pka_diff, acid_pka_diff

def construct_coupling_matrix(indices, state_strs, state_vecs, base_pka_diffs, acid_pka_diffs, coupling_cutoff=1.):
    coupling_matrix = np.zeros((len(indices),len(indices)))
    # print(indices)
    for state_str, state_vec in zip(state_strs[1:], state_vecs[1:]):
        changed_rel_idx = np.where(state_vec != 1)[0][0]
        coupling_matrix[changed_rel_idx] += np.where(base_pka_diffs[state_str] >= coupling_cutoff, 1, 0)
        coupling_matrix[changed_rel_idx] += np.where(acid_pka_diffs[state_str] >= coupling_cutoff, 1, 0)
    return coupling_matrix

def cluster_coupling_matrix(M):
    n = M.shape[0]
    visited = set()
    clusters = []

    def dfs(i, cluster):
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
    clusters = [list(c) for c in clusters]
    return clusters