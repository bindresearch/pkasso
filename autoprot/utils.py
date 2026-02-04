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