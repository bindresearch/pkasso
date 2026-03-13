import numpy as np

def pack_vec(state_vec):
    state_str = "".join([str(x) for x in state_vec])
    return state_str

def unpack_vec(state_str):
    state_vec = np.array([int(s) for s in state_str],dtype=int)
    return state_vec

def calc_state_strs(state_vecs):
    state_strs = []
    for state_vec in state_vecs:
        state_str = pack_vec(state_vec)
        state_strs.append(state_str)
    return state_strs

def calc_qs_all(state_vecs):
    qs_all = []
    for state_vec in state_vecs:
        qs = state_vec - 1
        qs_all.append(qs)
    return qs_all

def get_atom_with_map_idx(mol, map_idx):
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_idx:
            return atom
    return None

def sort_string(string,ps):
    """ Sort string by custom indices ps """

    s = list(string)
    s = [s[p] for p in ps]
    s = "".join(s)
    return s

def pack_indices(indices):
    indices_str = ''
    for id in indices:
        indices_str += f'{id},'
    # indices_str shows what atoms the cluster state_vec and state_str refer to
    indices_str = indices_str[:-1] # remove last comma
    return indices_str