import numpy as np

def construct_state_vectors_single(indices: list[int], q_options: np.ndarray) -> list[np.ndarray]:
    """
    Construct single-site perturbation state vectors.

    Generates protonation state vectors used for coupling analysis.
    The neutral reference state (all sites neutral) is included, along
    with states where exactly one site is protonated or deprotonated,
    provided that transition is allowed by `q_options`.

    Returns
    -------
    state_vecs : list[np.ndarray]
        List of protonation state vectors.
    """

    state_vecs = []
    state_vec = np.ones((len(indices)),dtype=int) #[1 for _ in indices]
    state_vecs.append(state_vec)

    for rel_idx, map_idx in enumerate(indices):
        for q in [0,2]:
            if q_options[rel_idx][q] == 1:
                state_vec = np.ones((len(indices)),dtype=int) #[1 for _ in indices]
                state_vec[rel_idx] = q
                state_vecs.append(state_vec)
    return state_vecs

def compare_pkas(
    indices: list[int],
    q_options: np.ndarray,
    state_str0: str,
    state_str1: str,
    base_lib: dict[str, dict[int, float]],
    acid_lib: dict[str, dict[int, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pKa differences between two protonation states.

    For each site, this function compares predicted acid and base pKa
    values between a reference state (`state_str0`) and a perturbed
    state (`state_str1`). Missing predictions are treated as large
    differences to indicate strong coupling.

    Returns
    -------
    base_pka_diff : np.ndarray
        Absolute differences in predicted base pKa values per site.
    acid_pka_diff : np.ndarray
        Absolute differences in predicted acid pKa values per site.
    """

    base_pka_diff = np.zeros((len(indices)))
    acid_pka_diff = np.zeros((len(indices)))
    for rel_idx, at_idx in enumerate(indices):
        if (q_options[rel_idx][2] == 1): # allowed for base
            if (at_idx in base_lib[state_str0]) and (at_idx in base_lib[state_str1]):
                base_pka_diff[rel_idx] = abs(base_lib[state_str1][at_idx] - base_lib[state_str0][at_idx])
            else:
                base_pka_diff[rel_idx] = 10. # one disappeared
        if (q_options[rel_idx][0] == 1): # allowed for acid
            if (at_idx in acid_lib[state_str0]) and (at_idx in acid_lib[state_str1]):
                acid_pka_diff[rel_idx] = abs(acid_lib[state_str1][at_idx] - acid_lib[state_str0][at_idx])
            else:
                acid_pka_diff[rel_idx] = 10. # one disappeared
    return base_pka_diff, acid_pka_diff

def construct_coupling_matrix(
    indices: list[int],
    state_strs: list[str],
    state_vecs: list[np.ndarray],
    base_pka_diffs: dict[str, np.ndarray],
    acid_pka_diffs: dict[str, np.ndarray],
    coupling_cutoff: float
    ) -> np.ndarray:
    """
    Build a site-site coupling matrix from pKa perturbations.

    The matrix records how strongly protonating or deprotonating one
    site affects the predicted pKa values of other sites. Entries are
    incremented when pKa differences exceed the specified cutoff.

    Returns
    -------
    coupling_matrix : np.ndarray
        Square matrix indicating pairwise coupling strength between sites.
    """

    coupling_matrix = np.zeros((len(indices),len(indices)))
    # print(indices)
    for state_str, state_vec in zip(state_strs[1:], state_vecs[1:]):
        changed_rel_idx = np.where(state_vec != 1)[0][0]
        coupling_matrix[changed_rel_idx] += np.where(base_pka_diffs[state_str] >= coupling_cutoff, 1, 0)
        coupling_matrix[changed_rel_idx] += np.where(acid_pka_diffs[state_str] >= coupling_cutoff, 1, 0)
    return coupling_matrix

def cluster_coupling_matrix(M: np.ndarray) -> list[list[int]]:
    """
    Partition sites into clusters based on coupling connectivity.

    Identify connected components in the coupling matrix, grouping
    sites that influence each other into clusters.

    Returns
    -------
    clusters : list[list[int]]
        Lists of site indices belonging to each coupling cluster.
    """

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