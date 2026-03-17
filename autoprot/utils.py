import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Atom

def pack_vec(state_vec: np.ndarray) -> str:
    """ Pack vector into string. """
    state_str = "".join([str(x) for x in state_vec])
    return state_str

def unpack_vec(state_str: str) -> np.ndarray:
    """ Unpack string into vector. """
    state_vec = np.array([int(s) for s in state_str],dtype=int)
    return state_vec

def calc_state_strs(state_vecs: list[np.ndarray]) -> list[str]:
    """ Calc state strings from vectors. """
    state_strs = []
    for state_vec in state_vecs:
        state_str = pack_vec(state_vec)
        state_strs.append(state_str)
    return state_strs

def calc_qs_all(state_vecs: list[np.ndarray]) -> list[np.ndarray]:
    """ Convert state vectors into a vector of charges. """
    qs_all = []
    for state_vec in state_vecs:
        qs = state_vec - 1
        qs_all.append(qs)
    return qs_all

def get_atom_with_map_idx(mol: Mol, map_idx: int) -> Atom | None:
    """ Find atom of rdkit Mol object with specific map index. """
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == map_idx:
            return atom
    return None

def sort_string(string: str, ps: np.ndarray) -> str:
    """ Sort string by custom indices ps. """

    s = list(string)
    s = [s[p] for p in ps]
    s_out = "".join(s)
    return s_out

def pack_indices(indices: list[int]) -> str:
    """ Convert list of indices into comma-separated string. """
    indices_str = ''
    for id in indices:
        indices_str += f'{id},'
    indices_str = indices_str[:-1] # remove last comma
    return indices_str