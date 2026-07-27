"""Utility helpers for protonation state processing."""

from pathlib import Path
from typing import cast
import copy

import numpy as np
from numpy.typing import NDArray
from rdkit.Chem.rdchem import Atom, Mol
from rdkit import Chem



def construct_mol(mol0: Mol, indices: list[int], state_vec: NDArray[np.int64]) -> Mol:
    """
    Construct a protonation-state-specific molecule from a reference molecule.

    The function applies the protonation/deprotonation state encoded in
    ``state_vec`` to the atoms specified by ``indices`` (atom map numbers).
    Formal charges are adjusted accordingly and hydrogens are added or removed
    where required. The resulting molecule is sanitized and returned.

    Parameters
    ----------
    mol0
        Reference molecule (neutral standardized structure)
        with atom map numbers assigned.
    indices
        Atom map indices corresponding to the sites whose states are
        defined in ``state_vec``.
    state_vec
        Protonation state vector for the selected sites. Values are encoded
        as [0, 1, 2] corresponding to [deprotonated, unchanged, protonated].

    Returns
    -------
    mol
        RDKit molecule with the specified protonation states applied.
    """

    mol_cand = copy.deepcopy(mol0)

    qs = state_vec - 1

    rw = Chem.RWMol(Chem.AddHs(mol_cand))

    for map_idx, q in zip(indices, qs):
        atom = get_atom_with_map_idx(rw, map_idx)
        if atom is None:
            raise ValueError(f"Could not find atom with map index {map_idx}.")
        atom.SetFormalCharge(int(q))
        if q == -1:
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    rw.RemoveAtom(nbr.GetIdx())
                    break

    mol_cand = Chem.RemoveHs(rw)
    Chem.SanitizeMol(mol_cand)

    return mol_cand

def pack_vec(state_vec: NDArray[np.int64]) -> str:
    """Pack vector into string."""

    state_str = "".join([str(x) for x in state_vec])
    return state_str

def unpack_vec(state_str: str) -> NDArray[np.int64]:
    """Unpack string into vector."""

    state_vec = np.array([int(s) for s in state_str], dtype=int)
    return state_vec

def calc_state_strs(state_vecs: list[NDArray[np.int64]]) -> list[str]:
    """Calc state strings from vectors."""

    state_strs = []
    for state_vec in state_vecs:
        state_str = pack_vec(state_vec)
        state_strs.append(state_str)
    return state_strs

def get_atom_with_map_idx(mol: Mol, map_idx: int) -> Atom | None:
    """Find atom of rdkit Mol object with specific map index."""

    for atom in cast(list[Atom], mol.GetAtoms()):
        if atom.GetAtomMapNum() == map_idx:
            return atom
    return None


def sort_string(string: str, ps: NDArray[np.int64]) -> str:
    """Sort string by custom indices ps."""

    s = list(string)
    s = [s[p] for p in ps]
    s_out = "".join(s)
    return s_out


def pack_indices(indices: list[int]) -> str:
    """Convert list of indices into comma-separated string."""

    indices_str = ""
    for id in indices:
        indices_str += f"{id},"
    indices_str = indices_str[:-1]  # remove last comma
    return indices_str

def is_jupyter() -> bool:
    """Return whether the code is running in a Jupyter kernel."""

    from collections.abc import Mapping
    from importlib import import_module

    try:
        ipython = import_module("IPython")
    except ImportError:
        return False

    get_ipython = getattr(ipython, "get_ipython", None)
    if not callable(get_ipython):
        return False

    shell = get_ipython()
    if shell is None:
        return False

    config = cast(Mapping[str, object], getattr(shell, "config", {}))
    return "IPKernelApp" in config

def state_str_to_q(state_str: str) -> str:
    """Convert state_str (0, 1, 2) to
    string of charges (-, 0, +)"""

    state_str_to_q_dict = {
        "0": "-",
        "1": "0",
        "2": "+",
    }

    q = ""
    for s in state_str:
        q += state_str_to_q_dict[s]
    return q


#### INPUT / OUTPUT ####


def read_smi(smi: Path) -> dict[str, str]:
    """Parse input .smi files"""

    batch_dict: dict[str, str] = {}

    ct = 0

    with open(smi, "r") as f:
        for line in f.readlines():
            spl = line.split()
            if len(spl) > 1:
                batch_dict[spl[1]] = spl[0]
            else:
                batch_dict[f'molecule{ct}'] = spl[0]
                ct += 1
    return batch_dict
