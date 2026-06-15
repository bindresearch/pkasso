from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from numpy.typing import NDArray
from torch_geometric.data import Data

ACCEPTOR_SMARTS_ONE = "[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]"
ACCEPTOR_SMARTS_TWO = "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"  # noqa: E501
DONOR_SMARTS_ONE = "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]"
DONOR_SMARTS_TWO = "[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]"

HYDROGEN_DONOR_ONE = Chem.MolFromSmarts(DONOR_SMARTS_ONE)
HYDROGEN_DONOR_TWO = Chem.MolFromSmarts(DONOR_SMARTS_TWO)
HYDROGEN_ACCEPTOR_ONE = Chem.MolFromSmarts(ACCEPTOR_SMARTS_ONE)
HYDROGEN_ACCEPTOR_TWO = Chem.MolFromSmarts(ACCEPTOR_SMARTS_TWO)

ATOM_SYMBOLS = ["C", "H", "O", "N", "S", "Cl", "F", "Br", "P", "I"]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def one_hot(x: Any, allowable_set: list[Any]) -> list[bool]:
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_bond_pair(mol: Mol) -> list[list[int]]:
    bonds = mol.GetBonds()
    res: list[list[int]] = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def _match_tuples(mol: Mol, query_one: Mol, query_two: Mol) -> set[tuple[int, ...]]:
    matches = set(mol.GetSubstructMatches(query_one))
    matches.update(mol.GetSubstructMatches(query_two))
    return matches


class MolVectorizer:
    """Cache molecule-level descriptors for repeated target-atom evaluations."""

    def __init__(self, mol: Mol) -> None:
        self.mol = mol
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol)

        self.n_atoms = mol.GetNumAtoms()
        self.edge_index = torch.tensor(get_bond_pair(mol), dtype=torch.long)
        self.batch = torch.zeros(self.n_atoms, dtype=torch.long)
        self.ring = mol.GetRingInfo()
        self.hydrogen_donor_matches = _match_tuples(mol, HYDROGEN_DONOR_ONE, HYDROGEN_DONOR_TWO)
        self.hydrogen_acceptor_matches = _match_tuples(mol, HYDROGEN_ACCEPTOR_ONE, HYDROGEN_ACCEPTOR_TWO)
        self.base_node_features = self._calc_base_node_features()
        self.shortest_path_lengths_by_aid: dict[int, NDArray[np.int64]] = {}

    def _calc_base_node_features(self) -> list[list[Any]]:
        features = []
        for atom_idx in range(self.n_atoms):
            atom = self.mol.GetAtomWithIdx(atom_idx)

            atom_features: list[Any] = []
            atom_features += one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
            atom_features += [atom.GetDegree()]
            atom_features += one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
            atom_features += [atom.GetValence(Chem.ValenceType.IMPLICIT)]
            atom_features += [atom.GetIsAromatic()]
            atom_features += [
                self.ring.IsAtomInRingOfSize(atom_idx, 3),
                self.ring.IsAtomInRingOfSize(atom_idx, 4),
                self.ring.IsAtomInRingOfSize(atom_idx, 5),
                self.ring.IsAtomInRingOfSize(atom_idx, 6),
                self.ring.IsAtomInRingOfSize(atom_idx, 7),
                self.ring.IsAtomInRingOfSize(atom_idx, 8),
            ]
            atom_features += [atom_idx in self.hydrogen_donor_matches]
            atom_features += [atom_idx in self.hydrogen_acceptor_matches]
            atom_features += [atom.GetFormalCharge()]
            features.append(atom_features)
        return features

    def _get_shortest_path_lengths(self, aid: int) -> NDArray[np.int64]:
        if aid in self.shortest_path_lengths_by_aid:
            return self.shortest_path_lengths_by_aid[aid]

        lengths = np.zeros(self.n_atoms, dtype=np.int64)
        for atom_idx in range(self.n_atoms):
            if atom_idx != aid:
                lengths[atom_idx] = len(Chem.rdmolops.GetShortestPath(self.mol, atom_idx, aid))
        self.shortest_path_lengths_by_aid[aid] = lengths
        return lengths

    def get_atom_features(self, aid: int) -> list[list[Any]]:
        shortest_path_lengths = self._get_shortest_path_lengths(aid)
        features = []
        for atom_idx, base_features in enumerate(self.base_node_features):
            features.append(
                [
                    *base_features,
                    int(shortest_path_lengths[atom_idx]),
                    atom_idx == aid,
                ]
            )
        return features

    def mol2vec(
        self,
        atom_idx: int,
        evaluation: bool = True,
        pka: float | None = None,
    ) -> Data:
        node_f = self.get_atom_features(atom_idx)
        x = torch.tensor(node_f, dtype=torch.float32)
        if evaluation:
            data = Data(
                x=x,
                edge_index=self.edge_index,
                batch=self.batch,
            )
        else:
            data = Data(
                x=x,
                edge_index=self.edge_index,
                pka=torch.tensor([[pka]], dtype=torch.float),
            )
        return data


def get_atom_features(mol: Mol, aid: int) -> list[list[Any]]:
    AllChem.ComputeGasteigerCharges(mol)
    Chem.AssignStereochemistry(mol)

    hydrogen_donor_matches = _match_tuples(mol, HYDROGEN_DONOR_ONE, HYDROGEN_DONOR_TWO)
    hydrogen_acceptor_matches = _match_tuples(mol, HYDROGEN_ACCEPTOR_ONE, HYDROGEN_ACCEPTOR_TWO)

    ring = mol.GetRingInfo()

    m = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)

        o: list[Any] = []
        o += one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
        o += [atom.GetDegree()]
        o += one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        o += [atom.GetValence(Chem.ValenceType.IMPLICIT)]
        o += [atom.GetIsAromatic()]
        o += [
            ring.IsAtomInRingOfSize(atom_idx, 3),
            ring.IsAtomInRingOfSize(atom_idx, 4),
            ring.IsAtomInRingOfSize(atom_idx, 5),
            ring.IsAtomInRingOfSize(atom_idx, 6),
            ring.IsAtomInRingOfSize(atom_idx, 7),
            ring.IsAtomInRingOfSize(atom_idx, 8),
        ]

        o += [atom_idx in hydrogen_donor_matches]
        o += [atom_idx in hydrogen_acceptor_matches]
        o += [atom.GetFormalCharge()]
        if atom_idx == aid:
            o += [0]
        else:
            o += [len(Chem.rdmolops.GetShortestPath(mol, atom_idx, aid))]

        if atom_idx == aid:
            o += [True]
        else:
            o += [False]
        m.append(o)
    return m


def mol2vec(
    mol: Mol,
    atom_idx: int,
    evaluation: bool = True,
    pka: float | None = None,
) -> Data:
    return MolVectorizer(mol).mol2vec(atom_idx, evaluation=evaluation, pka=pka)
