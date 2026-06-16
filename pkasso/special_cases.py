"""Special-case handling for protonation state generation."""
# mypy: disable-error-code=no-untyped-call

import logging
from collections import deque

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import Atom, Mol

from .utils import get_atom_with_map_idx

logger = logging.getLogger(__name__)


def match_smarts(mol: Mol, smarts: str) -> tuple[tuple[int, ...], ...]:
    """Match smarts pattern in rdkit molecule.

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
    assert pattern is not None, f"Invalid SMARTS pattern: {smarts}"

    matches: tuple[tuple[int, ...], ...] = mol.GetSubstructMatches(pattern)
    return matches


def find_charged(mol: Mol) -> list[int]:
    """
    Find map indices that could not be neutralized during preprocessing.
    These are removed from consideration.
    """

    mol_h = Chem.rdmolops.AddHs(mol)
    q0s = [at.GetFormalCharge() for at in mol.GetAtoms()]

    charged_indices = []
    for at_idx, q in enumerate(q0s):
        atom = mol_h.GetAtomWithIdx(at_idx)
        map_idx = atom.GetAtomMapNum()

        if q != 0:
            logger.info(f"Input molecule is charged at idx {at_idx}, map_idx {map_idx}!")
            charged_indices.append(map_idx)

    return charged_indices


def add_exclusion(exclusion_ids: set[int], mol: Mol, atom: Atom, smarts: str) -> set[int]:
    """
    Add simple q_options exclusion based on smarts pattern. Adds to previous exclusion_ids set.
    """

    map_idx = atom.GetAtomMapNum()
    matches = match_smarts(mol, smarts)

    for mat in matches:
        if atom.GetIdx() in mat:
            exclusion_ids.add(map_idx)

    return exclusion_ids


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
    matches = match_smarts(mol, smarts)

    phosphate_groups: dict[int, list[int]] = {}

    for mat in matches:
        p_map_idx: int | None = None

        # Find central P of phosphate
        for idx in mat:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "P":
                p_map_idx = atom.GetAtomMapNum()
                if p_map_idx not in phosphate_groups:
                    phosphate_groups[p_map_idx] = []

        assert p_map_idx is not None, "Phosphate SMARTS match did not include a phosphorus atom"

        # Find protonable O of phosphate
        for idx in mat:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "O" and (atom.GetTotalNumHs() > 0 or atom.GetFormalCharge() == -1):
                oh_map_idx = atom.GetAtomMapNum()
                if oh_map_idx not in phosphate_groups[p_map_idx]:
                    phosphate_groups[p_map_idx].append(oh_map_idx)

    found = len(matches) > 0
    return found, phosphate_groups


def short_alkyl(n_atom: Atom) -> bool:
    n_idx = n_atom.GetIdx()

    visited: set[int] = {n_idx}
    queue: deque[tuple[Atom, int]] = deque([(n_atom, 0)])

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


def has_invalid_amine(mol: Mol) -> int:
    """Special case amine with only ethyl or methyl or isopropyl (or no) substituents
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

        map_idx: int = atom.GetAtomMapNum()

        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                continue
            if nbr.GetAtomicNum() != 6:
                return 0

            if not short_alkyl(atom):
                return 0
        return map_idx
    return 0


def oh_ring_sulfonate(mol: Mol) -> list[int]:
    """
    Return atom indices of hydroxyl oxygens attached to the same fused/conjugated
    aromatic ring system as a sulfonate/sulfonic acid substituent.
    """

    phenol_pat = Chem.MolFromSmarts("[c][OX2H]")  # aromatic OH
    sulfo_pat = Chem.MolFromSmarts("[c]S(=O)(=O)[OX1H0-,OX2H1]")  # sulfonic acid / sulfonate
    assert phenol_pat is not None
    assert sulfo_pat is not None

    aromatic_rings: list[set[int]] = [
        set(ring) for ring in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    ]

    # Merge fused aromatic rings into aromatic ring systems
    systems: list[set[int]] = []
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
        new_systems: list[set[int]] = []
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

    sulfo_ring_atoms = {mat[0] for mat in mol.GetSubstructMatches(sulfo_pat)}

    matching_oxygen_indices: set[int] = set()

    for aromatic_atom_idx, oxygen_idx in phenol_matches:
        for system in systems:
            if aromatic_atom_idx in system and system & sulfo_ring_atoms:
                matching_oxygen_indices.add(oxygen_idx)
                break

    return sorted(matching_oxygen_indices)


def has_nplus_base_proximity(map_idx: int, mol: Mol, max_distance: int = 3) -> int:
    """
    Detect whether any neutral/basic nitrogen is within
    <= max_distance bonds of a positively charged nitrogen.
    """

    # Collect atom indices
    cation_nitrogens: list[int] = []

    atom0 = get_atom_with_map_idx(mol, map_idx)
    assert atom0 is not None
    at_idx = atom0.GetIdx()

    if atom0.GetAtomicNum() != 7:
        return 0
    aromatic0 = atom0.GetIsAromatic()

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue

        charge = atom.GetFormalCharge()
        aromatic = atom.GetIsAromatic()
        # positively charged nitrogen
        if charge > 0:
            cation_nitrogens.append(atom.GetIdx())
            if aromatic:
                cation_nitrogens.append(atom.GetIdx())  # add again for aromatics
            if aromatic0:
                cation_nitrogens.append(atom.GetIdx())  # add again for aromatics

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
