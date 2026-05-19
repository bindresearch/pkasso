"""Special-case handling for protonation state generation."""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Atom
from rdkit.Chem import rdmolops
from collections import deque

from .transitions import calc_freqs_from_states, calc_state_diffs
from .utils import unpack_vec, get_atom_with_map_idx

logger = logging.getLogger(__name__)

def match_smarts(mol: Mol, smarts: str) -> tuple[tuple[int]]:
    """ Match smarts pattern in rdkit molecule.
    
    Parameters:
    -----------
    mol
        Rdkit molecule
    smarts
        smiles string to match.

    Returns
    -------
    found
        Boolean returning if at least one match was found.
    matches
        Atom indices for each match.
    """

    pattern = Chem.MolFromSmarts(smarts)

    # found = mol.HasSubstructMatch(pattern)
    matches = mol.GetSubstructMatches(pattern)
    return matches

def find_charged(mol: Mol) -> list[int]:
    """
    Find map indices that could not be neutralized during preprocessing.
    These are removed from consideration.
    """

    mol_h = Chem.rdmolops.AddHs(mol)
    q0s = np.array([at.GetFormalCharge() for at in mol.GetAtoms()]) # type: ignore

    charged_indices = []
    for at_idx, q in enumerate(q0s):
        atom = mol_h.GetAtomWithIdx(at_idx) 
        map_idx = atom.GetAtomMapNum()

        if q != 0:
            logger.info(f'Input molecule is charged at idx {at_idx}, map_idx {map_idx}!')
            charged_indices.append(map_idx)

    return charged_indices

def add_exclusion(exclusion_ids: set, mol: Mol, atom: Atom, smarts: str) -> set[int]:
    """
    Add simple q_options exclusion based on smarts pattern. Adds to previous exclusion_ids set.
    """

    map_idx = atom.GetAtomMapNum()
    matches = match_smarts(mol, smarts)

    for mat in matches:
        if atom.GetIdx() in mat:
            exclusion_ids.add(map_idx)
    
    return exclusion_ids

def add_exclusions(mol: Mol) -> tuple[list[int], list[int]]:
    """ 
    Exclusions act on the q_options level. Exclusions are removed from consideration
    for protonation/deprotonation. This is specified separately for acids and bases.
    For example, only a protonation event could be excluded for a given index,
    but not the deprotonation event.
    This does not affect the indices or cluster splitting/
    
    Parameters
    ----------
    mol
        Input molecule with atom mapping numbers used to track sites.

    Returns
    -------
    exclude_base_indices
        Atom map indices for which protonation (base behavior) should
        be excluded.
    exclude_acid_indices
        Atom map indices for which deprotonation (acid behavior) should
        be excluded.
    """

    q0s = np.array([at.GetFormalCharge() for at in mol.GetAtoms()]) # type: ignore

    exclude_acid_indices: set[int] = set()
    exclude_base_indices: set[int] = set()
    # mol_h = Chem.rdmolops.AddHs(mol)

    smarts_imine = "NC(=N)"
    matches_imine = match_smarts(mol,smarts_imine)

    smarts_sulfonamide = "NS(=O)(=O)"

    smarts_diphenylamine = 'N(c)c'
    smarts_Ncnn = 'Nc(n)n'
    smarts_Nccn = 'Nc(c)n'
    smarts_Nccn2 = 'Nccn'
    smarts_nnn = 'nnn'
    smarts_ncnn = 'ncnn'
    smarts_cNO = 'C=NO'
    smarts_NNC = 'N-N=C'
    smarts_carbonyl = '[#7]~[#6X3](=[#8])'
    #
    smarts_ONphos = 'OC=NP(=O)(O)O'
    matches_ONphos = match_smarts(mol, smarts_ONphos)
    smarts_ONO = '[O]-[N+]([O-])'
    smarts_ONCO1 = 'O=N-C=O'
    smarts_ONCO2 = 'C=C(N=O)O'

    for at_idx, q in enumerate(q0s):
        atom = mol.GetAtomWithIdx(at_idx)
        map_idx = atom.GetAtomMapNum()

        if q != 0:
            continue
        if atom.GetSymbol() == 'O':
            for mat in matches_ONphos:
                if (atom.GetIdx() in mat):
                    correct_O = False
                    neighbors = atom.GetNeighbors()
                    for nbr in neighbors:
                        if nbr.GetSymbol() == 'C':
                            correct_O = True # O=CN part of the match
                    if correct_O:
                        exclude_acid_indices.add(map_idx)
            for smarts in [
                smarts_ONO,
                smarts_ONCO1,
                smarts_ONCO2
            ]:
                exclude_acid_indices = add_exclusion(exclude_acid_indices, mol, atom, smarts)

        if atom.GetSymbol() == 'N':
            # aromatic n
            if atom.GetIsAromatic():
                for smarts in [
                    smarts_carbonyl,
                    smarts_nnn,
                    smarts_ncnn
                ]:
                    exclude_base_indices = add_exclusion(exclude_base_indices, mol, atom, smarts)
                # ring Ns contributing to pi system
                if (atom.GetTotalNumHs() > 0) or (atom.GetDegree() == 3):
                    exclude_base_indices.add(map_idx)
            # non-aromatic N
            else:
                for smarts in [
                    smarts_sulfonamide,
                    smarts_diphenylamine,
                    smarts_Ncnn,
                    smarts_Nccn,
                    smarts_Nccn2,
                    smarts_cNO,
                    smarts_NNC
                ]:
                    exclude_base_indices = add_exclusion(exclude_base_indices, mol, atom, smarts)
                for mat in matches_imine: # ...N-C(=N)...
                    if atom.GetIdx() in mat:
                        accept = True
                        for bond in atom.GetBonds(): # Find the correct of the two Ns
                            if bond.GetBondType() == Chem.BondType.DOUBLE:
                                accept = False
                        if accept:
                            exclude_base_indices.add(map_idx)

    exclude_base_indices_sorted = sorted(exclude_base_indices)
    exclude_acid_indices_sorted = sorted(exclude_acid_indices)

    return exclude_base_indices_sorted, exclude_acid_indices_sorted

def has_phosphate(mol: Mol) -> tuple[bool, dict[int, list[int]]]:
    """
    Detect phosphate groups and their protonable hydroxyl atoms.

    Searches the molecule for phosphate motifs and returns the atom map
    indices of the central phosphorus atoms along with the indices of
    attached protonable oxygen atoms.

    Returns
    -------
    found
        True if at least one phosphate group is detected.
    phosphate_groups
        Mapping from phosphate P atom map indices to protonable OH
        atom map indices.
    """

    smarts = "P(=O)(O)(O)"
    matches = match_smarts(mol,smarts)

    phosphate_groups: dict[int, list[int]] = {}

    for mat in matches:
        # Find central P of phosphate
        for idx in mat:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "P":
                p_map_idx = atom.GetAtomMapNum()
                if p_map_idx not in phosphate_groups:
                    phosphate_groups[p_map_idx] = []
        # Find protonable O of phosphate
        for idx in mat:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "O" and (atom.GetTotalNumHs() > 0 or atom.GetFormalCharge() == -1.):
                oh_map_idx = atom.GetAtomMapNum()
                if oh_map_idx not in phosphate_groups[p_map_idx]: # type: ignore
                    phosphate_groups[p_map_idx].append(oh_map_idx) # type: ignore

    found = len(matches) > 0
    return found, phosphate_groups

def calc_phosphate_clusters(
    phosphate_groups: dict[int, list[int]],
    pH: float,
    matrix_def: str,
) -> tuple[list[list[str]], list[list[float]], list[list[int]]]:
    """
    Compute protonation state distributions for phosphate groups.

    Phosphates are treated as independent clusters with predefined
    pKa values. This function enumerates possible protonation states
    of the phosphate OH groups and calculates their equilibrium
    populations at the given pH.

    Returns
    -------
    state_strs_poh
        Protonation state encodings for each phosphate cluster.
    state_freqs_poh
        Corresponding microstate frequencies.
    oh_ids_poh
        Atom map indices of protonable OH atoms for each cluster.
    """

    state_strs_poh = []
    state_freqs_poh = []
    oh_ids_poh = []
    
    pka1 = 2.0
    pka2 = 6.5

    poh_acid_pkas_single: dict[str, list[float]] = {
        '0' : [pka1],
        '1' : [pka1],
    }

    base_lib_poh_single: dict[str, Any] = {
        '0': {},
        '1': {},
    }

    poh_acid_pkas_double: dict[str, list[float]] = {
        '00' : [pka2, pka2],
        '01' : [pka1, pka2],
        '10' : [pka2, pka1],
        '11' : [pka1, pka1],
    }

    base_lib_poh_double: dict[str, Any] = {
        '00': {},
        '01': {},
        '10': {},
        '11': {},
    }

    for p_idx, oh_ids in phosphate_groups.items():
        if len(oh_ids) == 1:
            state_strs = ['0','1']
            state_vecs = [unpack_vec(state_str) for state_str in state_strs]
            poh_acid_pkas = poh_acid_pkas_single
            base_lib_poh = base_lib_poh_single
        elif len(oh_ids) == 2:
            state_strs = ['00','01','10','11']
            state_vecs = [unpack_vec(state_str) for state_str in state_strs]
            poh_acid_pkas = poh_acid_pkas_double
            base_lib_poh = base_lib_poh_double
        else:
            logger.debug(f'Did not find protonable O for phosphate {p_idx}')
            continue
        acid_lib_poh: dict[str, dict[int, float]] = {}
        for key, val in poh_acid_pkas.items():
            acid_lib_poh[key] = {}
            for jdx, oh_id in enumerate(oh_ids):
                acid_lib_poh[key][oh_id] = val[jdx]

        logger.debug(oh_ids)
        logger.debug(acid_lib_poh)
        
        ps_all = calc_state_diffs(state_strs, state_vecs, oh_ids, base_lib_poh, acid_lib_poh, 
                                    pH=pH,matrix_def=matrix_def)
        
        state_strs_curated, state_freqs = calc_freqs_from_states(state_strs,state_vecs,ps_all,matrix_def)

        state_strs_poh.append(state_strs_curated)
        state_freqs_poh.append(state_freqs)
        oh_ids_poh.append(oh_ids)

    return state_strs_poh, state_freqs_poh, oh_ids_poh

def short_alkyl(n_atom, mol):
    n_idx = n_atom.GetIdx()

    visited = set([n_idx])
    queue = deque([(n_atom, 0)])

    max_dist = 0

    while queue:
        atom, dist = queue.popleft()

        for nbr in atom.GetNeighbors():
            idx = nbr.GetIdx()

            if idx == n_idx:
                continue

            # distance tracking
            if idx not in visited:
                visited.add(idx)

                # atom type constraint
                if nbr.GetAtomicNum() not in (1, 6):
                    return False

                # ignore hydrogens in distance logic
                if nbr.GetAtomicNum() == 1:
                    continue

                new_dist = dist + 1
                max_dist = max(max_dist, new_dist)

                if max_dist > 2:
                    return False

                queue.append((nbr, new_dist))

    return True


def has_invalid_amine(mol):
    """ Special case amine with only ethyl or methyl or isopropyl (or no) substituents
    e.g.
    CCNCC
    NCC
    NC
    CN(CC)CC
    CC(C)NC(C)C
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue

        map_idx = atom.GetAtomMapNum()

        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                continue
            if nbr.GetAtomicNum() != 6:
                return 0

            if not short_alkyl(atom, mol):
                return 0
        return map_idx
    return 0

def calc_single_fixed_pka(
    pH: float,
    pka: float,
    ss_lower: int,
    matrix_def: str,
) -> tuple[list[str], NDArray[np.float64]]:
    """ Fix the pka for an amine with only methyl or ethyl substituents (molgpka bug)"""

    state_strs = [f'{ss_lower}',f'{ss_lower+1}']
    state_vecs = [unpack_vec(state_str) for state_str in state_strs]

    # pka = 10.4 # rough average value from different short amines in IUPAC list

    base_lib: dict[str, dict[int, float]] = {
        f'{ss_lower}' : {0: pka},
        f'{ss_lower+1}' : {0: pka},
    }

    acid_lib: dict[str, dict[int, float]] = {
        f'{ss_lower}' : {},
        f'{ss_lower+1}' : {},
    }

    ids = [0] # pseudo index

    ps_all = calc_state_diffs(state_strs, state_vecs, ids, base_lib, acid_lib, 
                                    pH=pH,matrix_def=matrix_def)

    state_strs_curated, state_freqs = calc_freqs_from_states(state_strs,state_vecs,ps_all,matrix_def)

    return state_strs_curated, state_freqs

def oh_ring_sulfonate(mol) -> list[int]:
    """
    Return atom indices of hydroxyl oxygens attached to the same fused/conjugated
    aromatic ring system as a sulfonate/sulfonic acid substituent.
    """

    phenol_pat = Chem.MolFromSmarts("[c][OX2H]")  # aromatic OH
    sulfo_pat = Chem.MolFromSmarts("[c]S(=O)(=O)[OX1H0-,OX2H1]")  # sulfonic acid / sulfonate

    aromatic_rings = [
        set(ring)
        for ring in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    ]

    # Merge fused aromatic rings into aromatic ring systems
    systems = []
    for ring in aromatic_rings:
        merged = False
        for system in systems:
            if ring & system:
                system |= ring
                merged = True
                break
        if not merged:
            systems.append(set(ring))

    # Repeat merge in case ring A merged with B, then B with C
    changed = True
    while changed:
        changed = False
        new_systems = []
        while systems:
            system = systems.pop()
            overlaps = [s for s in systems if s & system]
            systems = [s for s in systems if not (s & system)]
            for s in overlaps:
                system |= s
                changed = True
            new_systems.append(system)
        systems = new_systems

    phenol_matches = [
        (mat[0], mat[1])  # aromatic atom, hydroxyl oxygen
        for mat in mol.GetSubstructMatches(phenol_pat)
    ]

    sulfo_ring_atoms = {
        mat[0]
        for mat in mol.GetSubstructMatches(sulfo_pat)
    }

    matching_oxygen_indices = set()

    for aromatic_atom_idx, oxygen_idx in phenol_matches:
        for system in systems:
            if aromatic_atom_idx in system and system & sulfo_ring_atoms:
                matching_oxygen_indices.add(oxygen_idx)
                break

    return sorted(matching_oxygen_indices)

def has_nplus_base_proximity(map_idx, mol, max_distance=3):
    """
    Detect whether any neutral/basic nitrogen is within
    <= max_distance bonds of a positively charged nitrogen.
    """

    # Collect atom indices
    cation_nitrogens = []

    atom0 = get_atom_with_map_idx(mol, map_idx)
    assert atom0 is not None
    at_idx = atom0.GetIdx()

    # atom0 = mol.GetAtomWithIdx(at_idx)
    if atom0.GetAtomicNum() != 7:
        return False
    aromatic0 = atom0.GetIsAromatic()
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        
        charge = atom.GetFormalCharge()
        aromatic = atom.GetIsAromatic()
        # positively charged nitrogen
        if (charge > 0):# and aromatic:
            cation_nitrogens.append(atom.GetIdx())
            if aromatic:
                cation_nitrogens.append(atom.GetIdx()) # add again for aromatics
            if aromatic0:
                cation_nitrogens.append(atom.GetIdx()) # add again for aromatics

    # If no candidates, exit early
    if not cation_nitrogens:
        return 0

    dist_matrix = rdmolops.GetDistanceMatrix(mol)

    count = 0
    # Check all pairs
    for c_idx in cation_nitrogens:
        if (c_idx != at_idx) and (dist_matrix[c_idx][at_idx] <= max_distance):
            # return True
            count += 1

    return count

