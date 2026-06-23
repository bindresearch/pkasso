"""Core pKasso workflow implementation."""

import copy
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, RegistrationHash
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Mol

from . import coupling, special_cases, utils
from .predict_pka import MolgpkaPredictor, Predictor
from .postprocess import Molecule, Scan, combine_results
from .transitions import calc_freqs_from_states, calc_state_diffs
from .utils import pack_indices, pack_vec, unpack_vec, state_str_to_q
from .tautomers import best_tautomer_smiles

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

def sizeable_organic_fragments(
        mol: Mol,
        min_heavy_atoms: int = 6
) -> list[dict[str,Any]]:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    sizeable = []
    for frag in frags:
        heavy_atoms = frag.GetNumHeavyAtoms()
        carbon_atoms = sum(1 for atom in frag.GetAtoms() if atom.GetAtomicNum() == 6)
        formal_charge = Chem.GetFormalCharge(frag)
        smiles = Chem.MolToSmiles(frag, canonical=True)

        if carbon_atoms > 0 and heavy_atoms >= min_heavy_atoms:
            sizeable.append(
                {
                    "smiles": smiles,
                    "heavy_atoms": heavy_atoms,
                    "carbon_atoms": carbon_atoms,
                    "formal_charge": formal_charge,
                    "mol_weight": Descriptors.ExactMolWt(frag),
                }
            )

    return sizeable

def preprocess(
    smiles_raw: str,
    tautomer_search: bool = False,
    max_tautomers: int = 100,
    num_confs: int = 10,
    strip_fragments: bool = True,
    score_window: int = 1,
) -> tuple[Mol, str]:
    """
    Construct and standardize an RDKit molecule from a SMILES string.
    Charges that cannot be neutralized (e.g., quaternary ammonium) are preserved.
    Atom map numbers are assigned to preserve mapping when the molecule gets changed
    (re-ordered, protonated, de-protonated).

    Parameters
    ----------
    smiles_raw
        Input SMILES string representing the molecule.
    tautomer_search
        Perform rough tautomer search
    Returns
    -------
    mol
        The standardized RDKit molecule with atom map numbers set to
        1-based indices.
    smiles
        Canonical SMILES representation of the processed molecule.
    """

    logger.debug("Raw:")
    logger.debug(smiles_raw)
    mol = Chem.MolFromSmiles(smiles_raw, sanitize=True)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles_raw}")
    
    if strip_fragments:
        sizeable = sizeable_organic_fragments(mol)
        if len(sizeable) > 1:
            raise ValueError(
                "Input SMILES contains multiple sizeable organic fragments:",
                sizeable
            )
        # Remove ions and covalent fragments
        chooser = rdMolStandardize.LargestFragmentChooser()
        mol = chooser.choose(mol)

    smiles = Chem.MolToSmiles(mol, canonical=True)

    logger.debug("Canonical")
    logger.debug(smiles)

    if tautomer_search:
        smiles = best_tautomer_smiles(
            smiles,
            max_tautomers=max_tautomers,
            num_confs=num_confs,
            score_window=score_window,
        )
    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    logger.debug("Formal charges before cleanup")
    charges = [at.GetFormalCharge() for at in mol.GetAtoms()]
    logger.debug(charges)

    mol = rdMolStandardize.Cleanup(mol)
    uncharger = rdMolStandardize.Uncharger(force=True)

    # load/save cycles to clean up the mol atom ordering
    mol = uncharger.uncharge(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    smiles = Chem.MolToSmiles(mol, canonical=True)

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    smiles = Chem.MolToSmiles(mol, canonical=True)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    logger.debug("Formal charges after cleanup")
    symbols = [at.GetFormalCharge() for at in mol.GetAtoms()]
    logger.debug(symbols)

    return mol, smiles


def find_candidate_sites(
    base: dict[int, float],
    acid: dict[int, float],
    exclude_base_indices: list[int],
    exclude_acid_indices: list[int],
    charged_indices: list[int],
) -> tuple[list[int], NDArray[np.int64]]:
    """
    Determine possible protonation and deprotonation sites for a molecule.
    Candidate atom indices are derived from predicted basic and acidic sites.

    Parameters
    ----------
    base
        Mapping of atom map indices to predicted basic pKa values.
    acid
        Mapping of atom map indices to predicted acidic pKa values.
    exclude_base_indices
        Atom map indices that must not be considered for protonation.
    exclude_acid_indices
        Atom map indices that must not be considered for deprotonation.
    charged_indices
        Atom map indices that could not be neutralized
    except_indices
        Atom map indices that require special treatment
    except_q_options
        q_options for except_indices

    Returns
    -------
    indices
        Sorted atom map indices considered for protonation state changes.
    q_options
        Array of shape (n_sites, 3) indicating allowed states per site:
        [deprotonated, unchanged, protonated].
    """

    prot_candidates = list(base.keys())  # should be map idx
    deprot_candidates = list(acid.keys())

    indices_raw = list(sorted(set(prot_candidates + deprot_candidates)))
    indices: list[int] = []

    logger.debug(f"relevant indices: {indices_raw}")

    # Remove indices for atoms that could not be neutralized
    for map_idx in indices_raw:
        if map_idx not in charged_indices:
            indices.append(map_idx)

    logger.debug(f"relevant indices (after charged removal): {indices}")

    q_options = np.zeros((len(indices), 3), dtype=np.int64)  # deprot=0, stay=1, prot=2
    for rel_idx, map_idx in enumerate(indices):
        q_options[rel_idx, 1] = 1  # always allow stay
        if map_idx in prot_candidates:
            if map_idx not in exclude_base_indices:
                q_options[rel_idx, 2] = 1  # allow protonation
        if map_idx in deprot_candidates:
            if map_idx not in exclude_acid_indices:
                q_options[rel_idx, 0] = 1  # allow deprotonation

    ps = np.argsort(indices)
    indices = [indices[p] for p in ps]
    q_options = q_options[ps]

    return indices, q_options


def construct_state_vectors(
    q_options: NDArray[np.int64],
    cutoff_states: int,
) -> list[NDArray[np.int64]]:
    """
    Enumerate all valid protonation state vectors given allowed site options.

    For each site, the allowed states are extracted from ``q_options`` and all
    combinations are generated via a Cartesian product. If the total number of
    possible combinations exceeds ``cutoff_states``, enumeration is skipped and
    an empty list is returned.

    Parameters
    ----------
    q_options
        Array of shape (n_sites, 3) indicating allowed states per site,
        where columns correspond to [deprotonated, unchanged, protonated]
        and entries are 1 (allowed) or 0 (disallowed).
    cutoff_states
        Maximum number of state combinations to enumerate.

    Returns
    -------
    state_vecs
        Array of shape (n_states, n_sites) containing all valid state vectors,
        or an empty list if the number of combinations exceeds ``cutoff_states``.
    """

    q_options_nonzero = []
    for rel_idx, qs in enumerate(q_options):
        q_col = []
        for q_idx, q in enumerate(qs):
            if q == 1.0:
                q_col.append(q_idx)
        if len(q_col) > 0.0:
            q_options_nonzero.append(q_col)

    N_trial_vecs = np.prod([len(qs) for qs in q_options_nonzero])
    if N_trial_vecs > cutoff_states:
        return []
    else:
        state_vecs = [np.array(x) for x in list(itertools.product(*q_options_nonzero))]
        return state_vecs


def count_state_combinations(q_options: NDArray[np.int64]) -> int:
    """Count valid protonation-state combinations without enumerating them."""

    q_counts = np.count_nonzero(q_options, axis=1)
    return int(np.prod(q_counts))


#########################################
# rdkit mol object construction


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
        atom = utils.get_atom_with_map_idx(rw, map_idx)
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


#############################################################################################
# Cluster tests and operations


@dataclass
class ProtonationIndexSpace:
    """pH-independent caches for one fixed protonation site space."""

    indices: list[int]
    q_options: NDArray[np.int64]
    mols_lib: dict[str, Mol] = field(default_factory=dict)
    base_lib: dict[str, dict[int, float]] = field(default_factory=dict)
    acid_lib: dict[str, dict[int, float]] = field(default_factory=dict)

    @property
    def indices_str(self) -> str:
        return pack_indices(self.indices)


@dataclass
class IndexSpaceRegistry:
    """Registry of pH-independent index spaces keyed by atom map indices."""

    spaces: dict[str, ProtonationIndexSpace] = field(default_factory=dict)

    def get_or_create(
        self,
        indices: list[int],
        q_options: NDArray[np.int64],
    ) -> ProtonationIndexSpace:
        indices_str = pack_indices(indices)
        if indices_str not in self.spaces:
            self.spaces[indices_str] = ProtonationIndexSpace(
                indices=list(indices),
                q_options=q_options.copy(),
            )
        space = self.spaces[indices_str]
        if not np.array_equal(space.q_options, q_options):
            raise ValueError(f"Conflicting q_options for indices {indices_str}")
        return space

    def get(self, indices: list[int]) -> ProtonationIndexSpace:
        return self.spaces[pack_indices(indices)]


@dataclass
class MicrostateDistribution:
    """pH-dependent microstate distribution over one fixed index space."""

    index_space: ProtonationIndexSpace
    pH: float
    state_strs: list[str]
    state_vecs: list[NDArray[np.int64]]
    state_freqs: list[float] | NDArray[np.float64]
    state_qs: dict[str, int] | None = None
    net_charge: float | None = None
    freqs_macro: dict[int, float] | None = None

    @property
    def indices(self) -> list[int]:
        return self.index_space.indices

    @property
    def mols_lib(self) -> dict[str, Mol]:
        return self.index_space.mols_lib

    def apply_symmetry(self) -> None:
        """Merge symmetry-equivalent states and keep state fields aligned."""

        state_freqs_by_state = {
            state_str: float(state_freq) for state_str, state_freq in zip(self.state_strs, self.state_freqs)
        }
        self.state_strs, self.state_freqs = calc_symmetry(
            self.state_strs,
            state_freqs_by_state,
            self.mols_lib,
        )
        self.state_vecs = [unpack_vec(state_str) for state_str in self.state_strs]

    def assign_macro_props(self) -> None:
        """Compute and store charge-resolved macrostate properties."""

        self.state_qs = calc_state_qs(self.state_strs, self.mols_lib)
        self.net_charge, self.freqs_macro = calc_macro_props(
            self.state_strs,
            self.state_freqs,
            self.state_qs,
        )


@dataclass
class PHScanDistribution:
    """Internal raw pH-scan distribution before public Scan postprocessing.

    This is the scan-level counterpart to MicrostateDistribution: it stores
    numerical pH-scan data produced by the core engine. The public Scan class
    in postprocess.py remains the outward-facing result with plotting and
    export helpers.
    """

    pHs: NDArray[np.float64]
    net_charges: list[float]
    state_freqs_all: dict[str, NDArray[np.float64]]
    freqs_macro_all: list[dict[int, float]]


def combine_cluster_distributions(
    cluster_dists: list[MicrostateDistribution],
    index_space: ProtonationIndexSpace,
    pH: float,
    sfreq_cutoff_combined: float = 0.01,
    max_states_combined: int = 100,
) -> MicrostateDistribution:
    """
    Combine microstate probabilities from independent pKa clusters.

    Microstates from different clusters are combined assuming statistical
    independence, with the combined microstate probability calculated as:

        p(AB) = p(A) * p(B)

    Prior to combination, low-frequency states are filtered within each
    cluster using ``sfreq_cutoff_individual``. After generating all possible
    combinations, a second filter is applied using
    ``sfreq_cutoff_combined``. Remaining probabilities are renormalized to
    sum to 1.

    Parameters
    ----------
    cluster_dists
        pH-dependent microstate distributions for independent clusters.
    index_space
        Combined protonation state space that the output distribution belongs to.
    pH
        Current pH value.
    sfreq_cutoff_individual
        Minimum relative frequency required for a state to be considered during
        cluster-wise filtering. Default is 0.01.
    sfreq_cutoff_combined
        Minimum relative frequency required for a combined microstate to be kept.
        Default is 0.001.
    max_states_combined
        Max. number of microstates at given pH value

    Returns
    -------
    microstate_distribution
        Combined microstate distribution over ``index_space``.
    """

    state_strs_clusters = [dist.state_strs for dist in cluster_dists]
    state_freqs_clusters = [np.asarray(dist.state_freqs, dtype=np.float64) for dist in cluster_dists]
    indices_clusters = [dist.indices for dist in cluster_dists]

    logger.debug(state_strs_clusters)
    logger.debug(state_freqs_clusters)

    cluster_state_ids = [range(len(state_freqs_cl)) for state_freqs_cl in state_freqs_clusters]
    n_combinations = int(np.prod([len(state_ids) for state_ids in cluster_state_ids]))
    logger.debug(f"N microstate combinations from clusters: {n_combinations}")

    indices = []
    for indices_cluster in indices_clusters:
        indices.extend(indices_cluster)  # This requires non-overlapping clusters!

    ps = np.argsort(indices)
    indices = [indices[p] for p in ps]
    indices_str = pack_indices(indices)
    if indices_str != index_space.indices_str:
        raise ValueError(f"indices_str {indices_str} not equal to full indices list {index_space.indices_str}")

    state_freq_max = float(np.prod([np.max(state_freqs_cl) for state_freqs_cl in state_freqs_clusters]))
    state_freq_cutoff = sfreq_cutoff_combined * state_freq_max
    sort_state_str = not np.array_equal(ps, np.arange(len(ps)))

    state_strs = []
    state_freqs_list = []

    combinations = itertools.product(*cluster_state_ids)
    for s_idxs in combinations:
        state_freq = 1.0
        state_str_parts = []
        for c_idx, s_idx in enumerate(s_idxs):
            state_str_parts.append(state_strs_clusters[c_idx][s_idx])
            state_freq *= float(state_freqs_clusters[c_idx][s_idx])

        if state_freq >= state_freq_cutoff:
            state_str = "".join(state_str_parts)
            if sort_state_str:
                state_str = utils.sort_string(state_str, ps)  # match sorted indices
            state_strs.append(state_str)
            state_freqs_list.append(state_freq)

    state_freqs = np.array(state_freqs_list)

    if len(state_strs) > max_states_combined:
        ps_state_strs = np.argsort(state_freqs)[::-1] # descending freq
        state_strs = [state_strs[p] for p in ps_state_strs][:max_states_combined]
        state_freqs = state_freqs[ps_state_strs][:max_states_combined]

    logger.debug(f"N chosen microstate combinations: {len(state_strs)}")
    # Correct freqs for removal of very unlikely states
    state_freqs /= np.sum(state_freqs)

    return MicrostateDistribution(
        index_space=index_space,
        pH=pH,
        state_strs=state_strs,
        state_vecs=[unpack_vec(state_str) for state_str in state_strs],
        state_freqs=state_freqs,
    )

def mol2hash(mol: Mol) -> str:
    """
    Generate a registration hash from an RDKit molecule.
    """

    return str(RegistrationHash.GetMolHash(RegistrationHash.GetMolLayers(mol)))


def calc_hashes(state_strs: list[str], mols_lib: dict[str, Mol]) -> list[str]:
    """
    Compute registration hashes for a set of microstates.

    Parameters
    ----------
    state_strs
        Microstate identifiers used as keys in ``mols_lib``.
    mols_lib
        Mapping from microstate strings to RDKit molecule objects.

    Returns
    -------
    hashes
        Registration hashes corresponding to the input microstates,
        in the same order as ``state_strs``.
    """

    hashes = []
    for state_str in state_strs:
        mol = mols_lib[state_str]
        hash = mol2hash(mol)
        hashes.append(hash)
    return hashes


def calc_symmetry(
    state_strs: list[str],
    state_freqs_lib: dict[str, float],
    mols_lib: dict[str, Mol],
) -> tuple[list[str], list[float]]:
    """
    Merge symmetry-equivalent microstates based on molecular hashes.

    Microstates that share the same registration hash are grouped together.
    Their probabilities are summed, and a single representative state
    (alphabetically sorted) is retained per group.

    Parameters
    ----------
    state_strs
        Microstate identifiers.
    state_freqs_lib
        Mapping from microstate strings to their probabilities.
    mols_lib
        Mapping from microstate strings to RDKit molecule objects.

    Returns
    -------
    state_strs_symm
        Symmetry-reduced microstate strings.
    state_freqs_symm
        Corresponding combined frequencies.
    """

    state_hashes = calc_hashes(state_strs, mols_lib)
    state_dict: dict[str, list[str]] = {}

    for state_str, state_hash in zip(state_strs, state_hashes):
        if state_hash in state_dict:
            state_dict[state_hash].append(state_str)
        else:
            state_dict[state_hash] = [state_str]

    logger.debug(state_dict)

    state_strs_symm: list[str] = []
    state_freqs_symm: list[float] = []

    for state_hash, state_strs_per_hash in state_dict.items():
        state_strs_sorted = sorted(state_strs_per_hash)
        state_strs_symm.append(state_strs_sorted[0])
        state_freq = 0.0
        for state_str in state_strs_per_hash:
            state_freq += state_freqs_lib[state_str]
        state_freqs_symm.append(state_freq)

    return state_strs_symm, state_freqs_symm


def calc_state_qs(
    state_strs: list[str],
    mols_lib: dict[str, Mol],
) -> dict[str, int]:
    """
    Compute formal charges for each microstate.

    Parameters
    ----------
    state_strs
        Microstate identifiers.
    mols_lib
        Mapping from microstate strings to RDKit molecule objects.

    Returns
    -------
    state_qs
        Dictionary mapping microstate strings to formal charges
    """

    state_qs: dict[str, int] = {}
    for state_str in state_strs:
        state_qs[state_str] = Chem.GetFormalCharge(mols_lib[state_str])
    return state_qs


def calc_macro_props(
    state_strs: list[str],
    state_freqs: list[float] | NDArray[np.float64],
    state_qs: dict[str, int],
) -> tuple[float, dict[int, float]]:
    """
    Compute macrostate properties from weighted microstate contributions.

    The net charge is calculated as the frequency-weighted sum of
    microstate formal charges. The charge-resolved frequency distribution
    is aggregated from the same per-microstate charges.

    Parameters
    ----------
    state_strs
        Microstate identifiers.
    state_freqs
        Corresponding normalized microstate probabilities.
    state_qs
        Mapping from microstate strings to formal charges.

    Returns
    -------
    net_charge
        Net charge (frequency-weighted sum)
    freqs_macro
        Dictionary mapping formal charges to their total frequency
    """

    freqs_macro: dict[int, float] = {}
    net_charge = 0.0
    for state_str, state_freq in zip(state_strs, state_freqs):
        state_q = state_qs[state_str]
        if state_q in freqs_macro:
            freqs_macro[state_q] += state_freq
        else:
            freqs_macro[state_q] = state_freq
        net_charge += state_q * state_freq
    return net_charge, freqs_macro


###########


def combine_pkas_macro(
    pHs: NDArray[np.float64],
    freqs_macro_all: list[dict[int, float]],
) -> dict[int, float]:
    """
    Estimate macrostate pKa values from charge-resolved frequency data.

    For each pH value, adjacent charge states (q and q+1) are used to
    compute a pKa estimate:

        pKa = log10(freq(q+1) / freq(q)) + pH

    Multiple pKa estimates for the same charge transition are combined
    using weighted averaging. pKa estimates from evaluations with pH values
    close to the pKa are weighted higher.

    Parameters
    ----------
    pHs
        Array of pH values corresponding to the frequency datasets.
    freqs_macro_all
        List of macrostate frequency dictionaries, one per pH value.
        Each dictionary maps formal charge (int) to its frequency.

    Returns
    -------
    pkas_combined
        Combined macrostate pKa values indexed by charge state.
        Represents pKa between q and q+1.
    """

    pkas_macro: dict[int, list[float]] = {}
    pkas_weights: dict[int, list[float]] = {}

    for pH, freqs_macro in zip(pHs, freqs_macro_all):
        qs_sorted = sorted(freqs_macro.keys())
        for q in qs_sorted:
            if q + 1 in qs_sorted:
                freq1 = freqs_macro[q]
                freq2 = freqs_macro[q + 1]
                pka_macro = np.log10(freq2 / freq1) + pH
                pka_weight = 1.0 / (freq1**2 + freq2**2)
                if q in pkas_macro:
                    pkas_macro[q].append(pka_macro)
                    pkas_weights[q].append(pka_weight)
                else:
                    pkas_macro[q] = [pka_macro]
                    pkas_weights[q] = [pka_weight]

    pkas_combined: dict[int, float] = {}

    for q, pkas in pkas_macro.items():
        ws = pkas_weights[q]
        pka_comb = float(np.average(pkas, weights=ws))
        pkas_combined[q] = pka_comb
    return pkas_combined


###########


@dataclass
class pKasso:
    """
    pKasso pipeline for protonation state prediction.

    Parameters
    ----------
    smiles_raw
        Input SMILES string of the molecule to be processed.
    **kwargs
        Optional configuration parameters. Supported keys include:

        Pipeline parameters:
            name, cutoff_states, device,
            pka_predictor_cls,
            sfreq_cutoff_individual, sfreq_cutoff_combined,
            matrix_def, cutoff_export

    """

    smiles: str
    name: str = "molecule"

    # Internal options
    cutoff_states: int = 200
    sfreq_cutoff_individual: float = 0.01
    max_states_individual: int = 20
    sfreq_cutoff_combined: float = 0.01
    max_states_combined: int = 20
    cutoff_export: float = 1.0
    matrix_def: str = "dG"
    device: str = "cpu"  # fixed!
    pka_predictor_cls: type[Predictor] = MolgpkaPredictor
    tautomer_search: bool = True
    max_tautomers: int = 20
    num_confs: int = 10
    total_max_sites: int = 25
    strip_fragments: bool = True
    score_window: int = 1

    def pka_predictor(self, mol: Mol) -> Predictor:
        """Create the configured molecule-specific pKa predictor."""

        return self.pka_predictor_cls(mol, device=self.device)

    def run_single(self, pH: float = 7.0) -> Molecule:
        """
        Run the full pKasso pipeline.

        1. Setup of models and preprocessing.
        2. pH-dependent state enumeration and thermodynamic evaluation.
        3. Final analysis, pKa aggregation, and visualization.
        """

        self.pH = pH
        self._setup()
        distribution = self._calc_microstates(self.pH)
        molecule = self.prep_single_output(distribution)
        return molecule

    def run_scan(
        self,
        pHs: NDArray[np.float64] = np.arange(0, 14.1, 0.5, dtype=np.float64),
    ) -> Scan:
        """
        Run pH scan
        """

        self._setup()
        distribution = self._scan_pH(pHs)
        return self._finalize_scan(distribution)

    #########################

    def _setup(self) -> None:
        """
        Initialize models, file structure, and neutral-state information.

        This step performs all non-pH-dependent initialization, including:

        - Creating output directories
        - Loading molgpka models
        - Preparing the neutral molecular state
        - Initializing result containers
        - Determining the full set of indices with protonable sites
        """

        logger.debug(self.name)
        logger.debug(self.smiles)

        self.initialize_paths_models_libs()

        self.mol0, self.smiles0 = preprocess(
            self.smiles,
            tautomer_search=self.tautomer_search,
            max_tautomers=self.max_tautomers,
            num_confs=self.num_confs,
            strip_fragments=self.strip_fragments,
            score_window=self.score_window,
        )

        self.charged_indices = special_cases.find_charged(self.mol0)
        pka_predictor = self.pka_predictor(self.mol0)
        self.exclude_base_indices, self.exclude_acid_indices = pka_predictor.exclude_sites()

        logger.debug("Processed:")
        logger.debug(self.smiles0)
        logger.debug(f"Exclude base indices: {self.exclude_base_indices}")
        logger.debug(f"Exclude acid indices: {self.exclude_acid_indices}")

        self.acid0 = pka_predictor.pred_acid()  # returns pkas for map indices
        self.base0 = pka_predictor.pred_base()  # returns pkas for map indices

        if len(self.acid0) + len(self.base0) > self.total_max_sites:
            raise ValueError(f'Molecule must contain <={self.total_max_sites} protonation sites.')

        self.indices0, self.q_options0 = find_candidate_sites(
            self.base0, self.acid0, self.exclude_base_indices, self.exclude_acid_indices, self.charged_indices
        )

        self.indices0_str = pack_indices(self.indices0)
        self.index_space0 = self.index_spaces.get_or_create(self.indices0, self.q_options0)

        # Screen coupling between residues, now pH independent
        self.clusters = self.screen_clusters(self.indices0, self.q_options0)
        self.cluster_spaces = [
            self.index_spaces.get_or_create(
                [self.indices0[c] for c in cluster],
                self.q_options0[cluster],
            )
            for cluster in self.clusters
        ]
        logger.debug(f"Clusters: {self.clusters}")

    def _calc_microstates(self, pH: float) -> MicrostateDistribution:
        """Calc microstate frequencies given a pH value"""

        # indices0_curated, q_options0 = self.calc_curated_indices(pH)

        cluster_dists: list[MicrostateDistribution] = []

        for cluster_space in self.cluster_spaces:
            cluster_dists.append(self.process_cluster(cluster_space, pH, 
                                                      sfreq_cutoff_individual=self.sfreq_cutoff_individual,
                                                      max_states_individual=self.max_states_individual))

        # Combine clusters and their frequencies
        dist = combine_cluster_distributions(
            cluster_dists,
            self.index_space0,
            pH,
            sfreq_cutoff_combined=self.sfreq_cutoff_combined,
            max_states_combined=self.max_states_combined,
        )

        self.construct_mols(dist.index_space, dist.state_strs, dist.state_vecs)

        # Symmetry (combine frequencies for chemically identical microstates)
        dist.apply_symmetry()

        # Macro-pka properties from combined microstates
        dist.assign_macro_props()

        return dist

    def _scan_pH(self, pHs: NDArray[np.float64]) -> PHScanDistribution:
        """
        Perform the full pH-dependent microstate enumeration and analysis.

        For each pH value in the configured pH grid, this method:

        - Identifies curated candidate titration sites
        - Screens residue coupling and builds clusters of coupled sites
        - Processes each cluster to generate microstates and frequencies
        - Optionally injects phosphate-specific clusters
        - Combines cluster results into global state distributions
        - Applies symmetry reduction
        - Computes macrostate properties (net charge and macro-pKa data)
        - Stores results for later visualization
        - Optionally exports pH-specific outputs
        """

        net_charges: list[float] = []
        state_freqs_all: dict[str, NDArray[np.float64]] = {}
        freqs_macro_all: list[dict[int, float]] = []

        for pH_idx, pH in enumerate(pHs.flat):
            distribution = self._calc_microstates(float(pH))

            if distribution.net_charge is None or distribution.freqs_macro is None:
                raise ValueError("Microstate distribution is missing macro properties.")
            net_charges.append(distribution.net_charge)
            freqs_macro_all.append(distribution.freqs_macro)

            # Add to results for pH scan
            for state_str, state_freq in zip(distribution.state_strs, distribution.state_freqs):
                if state_str not in state_freqs_all:
                    state_freqs_all[state_str] = np.zeros(len(pHs))
                state_freqs_all[state_str][pH_idx] = state_freq

        return PHScanDistribution(
            pHs=pHs,
            net_charges=net_charges,
            state_freqs_all=state_freqs_all,
            freqs_macro_all=freqs_macro_all,
        )

    def _finalize_scan(self, distribution: PHScanDistribution) -> Scan:
        """
        Post-process results, compute macro-pKa values, and generate outputs.

        This step performs final analysis and visualization, including:

        - Combining and exporting macro-pKa values across pH values
        - Identifying relevant microstates for plotting
        - Generating pH scan plots
        """

        net_charges = np.array(
            np.round(np.array(distribution.net_charges), decimals=4),
            dtype=np.float64,
        )

        pkas_macro = combine_pkas_macro(distribution.pHs, distribution.freqs_macro_all)

        state_strs_relevant: list[str] = []
        sfreqs_relevant: list[NDArray[np.float64]] = []
        mols_relevant: list[Mol] = []
        sfreqs_not_relevant: list[NDArray[np.float64]] = []

        if distribution.state_freqs_all:
            (
                state_strs_relevant,
                sfreqs_relevant,
                mols_relevant,
                sfreqs_not_relevant,
            ) = self.calc_relevant_states(distribution.state_freqs_all)

        return Scan(
            self.name,
            self.indices0,
            state_strs_relevant,
            mols_relevant,
            sfreqs_relevant,
            distribution.pHs,
            net_charges,
            sfreqs_not_relevant,
            pkas_macro,
        )

    def calc_relevant_states(
        self,
        state_freqs_all: dict[str, NDArray[np.float64]],
        max_states: int = 18,
    ) -> tuple[
        list[str],
        list[NDArray[np.float64]],
        list[Mol],
        list[NDArray[np.float64]],
    ]:
        """Reduce number of states to max_states for plotting."""

        cutoff = 0.01

        while True:
            state_strs_relevant: list[str] = []
            sfreqs_relevant: list[NDArray[np.float64]] = []
            sfreqs_not_relevant: list[NDArray[np.float64]] = []
            mols_relevant: list[Mol] = []
            pH_argmaxs: list[int] = []

            for state_str, sfreqs in state_freqs_all.items():
                mol = self.index_space0.mols_lib[state_str]
                mol.SetProp("_Name", state_str_to_q(state_str))
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)

                if np.max(sfreqs) > cutoff:
                    state_strs_relevant.append(state_str)
                    sfreqs_relevant.append(sfreqs)
                    mols_relevant.append(mol)
                    pH_argmaxs.append(int(np.argmax(sfreqs)))
                else:
                    sfreqs_not_relevant.append(sfreqs)

            N_relevant_states = len(state_strs_relevant)
            if N_relevant_states <= max_states:
                break

            cutoff += 0.02

        ps: list[int] = [int(p) for p in np.argsort(pH_argmaxs)]

        logger.debug(f"Final N relevant states: {N_relevant_states} with cutoff {cutoff}")
        return (
            [state_strs_relevant[p] for p in ps],
            [sfreqs_relevant[p] for p in ps],
            [mols_relevant[p] for p in ps],
            sfreqs_not_relevant,
        )

    #########################

    def initialize_paths_models_libs(self) -> None:
        """
        Reset internal libraries used to cache state-dependent predictions.
        """

        self.index_spaces = IndexSpaceRegistry()

    def process_cluster(
        self,
        space: ProtonationIndexSpace,
        pH: float,
        sfreq_cutoff_individual: float = 0.01,
        max_states_individual: int = 100,

    ) -> MicrostateDistribution:
        """
        Generate and evaluate microstates for a single protonation cluster at a given pH value.

        Parameters
        ----------
        space
            Fixed protonation site space for the current cluster.
        pH
            Current pH value in the pH scan.

        Returns
        -------
        microstate_distribution
            pH-dependent microstate distribution for this cluster.
        """

        state_vecs = construct_state_vectors(space.q_options, self.cutoff_states)

        state_strs = utils.calc_state_strs(state_vecs)

        self.construct_mols(space, state_strs, state_vecs)
        self.run_acid_base_calcs(space, state_strs, state_vecs)

        ps_all = calc_state_diffs(
            state_strs,
            state_vecs,
            space.indices,
            space.base_lib,
            space.acid_lib,
            pH=pH,
            matrix_def=self.matrix_def,
        )

        state_strs, state_freqs = calc_freqs_from_states(
            state_strs,
            state_vecs,
            ps_all,
            self.matrix_def,
        )

        # Cull

        ps = np.argsort(state_freqs)[::-1]
        state_strs = [state_strs[p] for p in ps][:max_states_individual]
        state_freqs = state_freqs[ps][:max_states_individual]

        max_state_freqs = np.max(state_freqs)

        state_strs_list = []
        state_freqs_list = []

        for state_str, state_freq in zip(state_strs, state_freqs):
            if (state_freq / max_state_freqs) >= sfreq_cutoff_individual:
                state_strs_list.append(state_str)
                state_freqs_list.append(state_freq)

        state_strs = state_strs_list
        state_freqs = np.array(state_freqs_list)

        state_vecs = [unpack_vec(state_str) for state_str in state_strs]
        return MicrostateDistribution(
            index_space=space,
            pH=pH,
            state_strs=state_strs,
            state_vecs=state_vecs,
            state_freqs=state_freqs,
        )

    #########################

    def coupling_assay_weights(
        self,
        indices: list[int],
        q_options: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """
        Perform pairwise pKa sensitivity analysis and return raw coupling weights.

        This method evaluates whether protonation of one site affects the
        predicted pKa values of other sites within the provided index set.

        Procedure:
        - Enumerate all single-site protonation states
        - Construct molecular representations for each state
        - Compare pKa values between reference and perturbed states
        - Build a pKa-difference weight matrix

        Parameters
        ----------
        indices
            Absolute atom map indices of the protonable sites being analyzed.
        q_options
            Array encoding allowed protonation states for each site.

        Returns
        -------
        coupling_weights
            Square matrix containing max acid/base pKa differences between sites.
        """

        space = self.index_spaces.get_or_create(indices, q_options)

        state_vecs = coupling.construct_state_vectors_single(indices, q_options)
        state_strs = utils.calc_state_strs(state_vecs)

        self.construct_mols(space, state_strs, state_vecs)
        self.run_acid_base_calcs(space, state_strs, state_vecs)

        for key, val in space.base_lib.items():
            logger.debug(f"{key}: {val}")

        state_str0 = state_strs[0]  # Neutral state
        base_pka_diffs = {}
        acid_pka_diffs = {}
        for state_str1 in state_strs[1:]:
            base_pka_diffs[state_str1], acid_pka_diffs[state_str1] = coupling.compare_pkas(
                indices, q_options, state_str0, state_str1, space.base_lib, space.acid_lib
            )

        return coupling.construct_coupling_weight_matrix(
            indices, state_strs, state_vecs, base_pka_diffs, acid_pka_diffs
        )

    # def coupling_assay_matrix(
    #     self,
    #     indices: list[int],
    #     q_options: NDArray[np.int64],
    #     coupling_cutoff: float,
    # ) -> NDArray[np.int64]:
    #     """
    #     Perform pairwise pKa sensitivity analysis and return a coupling matrix.

    #     This compatibility wrapper thresholds the raw pKa-difference weights at
    #     ``coupling_cutoff``.
    #     """

    #     coupling_weights = self.coupling_assay_weights(indices, q_options)
    #     return coupling.threshold_coupling_weights(coupling_weights, coupling_cutoff)

    def split_cluster_by_coupling_penalty(
        self,
        cluster: list[int],
        q_options0: NDArray[np.int64],
        coupling_weights: NDArray[np.float64],
        coupling_cutoff: float,
    ) -> list[list[int]]:
        """
        Recursively split an oversized cluster by local penalty-limited cuts.

        The cutoff is raised only for the subcluster currently being split. At
        each local cutoff, acceptable cuts sever at most two graph edges and
        have total pKa penalty no larger than 1.5 times the local cutoff.
        """

        graph = coupling.coupling_weights_to_graph(coupling_weights, coupling_cutoff, nodes=cluster)
        components = [sorted(component) for component in nx.connected_components(graph)]
        components = sorted(components, key=lambda c: c[0])

        if len(components) > 1:
            split_clusters = []
            for component in components:
                split_clusters.extend(
                    self.split_cluster_by_coupling_penalty(component, q_options0, coupling_weights, coupling_cutoff)
                )
            return split_clusters

        cluster = components[0] if components else sorted(cluster)
        if count_state_combinations(q_options0[cluster]) <= self.cutoff_states:
            return [cluster]

        child_clusters = coupling.find_best_penalty_limited_split(
            graph,
            coupling_weights,
            lambda child_cluster: count_state_combinations(q_options0[child_cluster]),
            max_cut_edges=2,
            coupling_cutoff=coupling_cutoff,
        )
        if child_clusters is not None:
            split_clusters = []
            for child_cluster in child_clusters:
                split_clusters.extend(
                    self.split_cluster_by_coupling_penalty(
                        child_cluster,
                        q_options0,
                        coupling_weights,
                        coupling_cutoff,
                    )
                )
            return split_clusters

        next_coupling_cutoff = round(coupling_cutoff + 0.1, 10)
        if next_coupling_cutoff > 1.5:
            logger.info(f"Local coupling cutoff high: {next_coupling_cutoff}")
        return self.split_cluster_by_coupling_penalty(
            cluster,
            q_options0,
            coupling_weights,
            next_coupling_cutoff,
        )

    def screen_clusters(self, indices0: list[int], q_options0: NDArray[np.int64]) -> list[list[int]]:
        """
        Determine stable pKa coupling clusters using adaptive thresholding.

        This method partitions protonable sites into independent clusters using
        an initial pKa coupling threshold. Oversized clusters are split
        recursively by applying the cheapest acceptable graph cut, with cutoff
        increases applied only to the subcluster currently being split.

        Stability criterion:
        A cluster is rejected if state enumeration exceeds the allowed
        cutoff, which indicates excessive coupling. In that case, candidate
        cutsets of one or two graph edges are considered. A cut is acceptable
        when the total severed pKa penalty is no larger than 1.5 times the
        local coupling cutoff. Among acceptable cuts, the split minimizing the
        summed child-cluster state count is selected.

        Parameters
        ----------
        indices0
            Absolute atom map indices of candidate protonation sites.
        q_options0
            Array encoding allowed protonation states for each site.

        Returns
        -------
        clusters
            Final set of stable coupling clusters.
        """

        coupling_cutoff = 0.1
        coupling_weights = self.coupling_assay_weights(indices0, q_options0)
        graph = coupling.coupling_weights_to_graph(coupling_weights, coupling_cutoff)
        clusters = [sorted(component) for component in nx.connected_components(graph)]

        split_clusters = []
        for cluster in sorted(clusters, key=lambda c: c[0]):
            split_clusters.extend(
                self.split_cluster_by_coupling_penalty(cluster, q_options0, coupling_weights, coupling_cutoff)
            )
        return split_clusters

    def construct_mols(
        self,
        space: ProtonationIndexSpace,
        state_strs: list[str],
        state_vecs: list[NDArray[np.int64]],
    ) -> None:
        """
        Construct and cache RDKit molecular objects for protonation states.

        For each unique protonation state defined by `state_strs` and
        `state_vecs`, this method:

        - Builds the corresponding RDKit molecule
        - Assigns the state string as the molecule name
        - Stores the molecule in the state-space cache

        Molecules are only constructed if they are not already present
        in the cache.

        Parameters
        ----------
        state_strs
            Encoded representations of protonation states.
        state_vecs
            Vector representations corresponding to `state_strs`.
        space
            Fixed protonation site space that owns the molecule cache.
        """

        for state_str, state_vec in zip(state_strs, state_vecs):
            if state_str not in space.mols_lib:
                mol_cand = construct_mol(self.mol0, space.indices, state_vec)
                mol_cand.SetProp("_Name", state_str)
                space.mols_lib[state_str] = mol_cand

    ###################################
    # Acid-base calculation

    def run_acid_base_calcs(
        self,
        space: ProtonationIndexSpace,
        state_strs: list[str],
        state_vecs: list[NDArray[np.int64]],
    ) -> None:
        """Compute and cache acid/base pKa predictions for microstates.

        For each protonation state, this method predicts site-specific
        acid and base pKa values using molgpka and stores the
        results in the state-space pKa caches.

        Predictions are evaluated from the neutral form of each site:
        - For base predictions, other sites are forced to at least neutral
        (deprotonations ignored).
        - For acid predictions, other sites are forced to at most neutral
        (protonations ignored).

        Parameters
        ----------
        state_strs
            Encoded representations of protonation states.
        state_vecs
            Protonation state vectors corresponding to `state_strs`.
        space
            Fixed protonation site space that owns the pKa caches.
        """

        for state_str, state_vec in zip(state_strs, state_vecs):
            if state_str in space.base_lib:
                continue

            logger.debug(state_str)

            state_vec_base = np.maximum(state_vec, 1)  # disregard de-protonations of other sites to assess base probability

            state_str_base = pack_vec(state_vec_base)

            mol_base = space.mols_lib[state_str_base]

            base_tmp = self.pka_predictor(mol_base).pred_base()
            base = {}
            for map_idx, b in base_tmp.items():
                if map_idx not in space.indices:
                    continue
                rel_idx = space.indices.index(map_idx)
                if (
                    state_vec[rel_idx] == 1
                ):  # Only consider predicted protonation/de-protonation predictions from neutral state
                    base[map_idx] = b

            state_vec_acid = np.minimum(state_vec, 1)  # disregard protonations of other sites to assess acid probability
            state_str_acid = pack_vec(state_vec_acid)

            mol_acid = space.mols_lib[state_str_acid]

            acid_tmp = self.pka_predictor(mol_acid).pred_acid()

            acid = {}
            for map_idx, a in acid_tmp.items():
                if map_idx not in space.indices:
                    continue
                rel_idx = space.indices.index(map_idx)
                if (
                    state_vec[rel_idx] == 1
                ):  # Only consider predicted protonation/de-protonation predictions from neutral state
                    acid[map_idx] = a

            space.base_lib[state_str] = base
            space.acid_lib[state_str] = acid

    def prep_single_output(
        self,
        distribution: MicrostateDistribution,
    ) -> Molecule:
        """
        Generate microstate output for the selected pH value.

        This method exports the most relevant protonation states for
        `pH`. States are selected based on their probability
        relative to the most populated state.

        The procedure includes:
        - Filtering and sorting microstates by probability
        - Checking stereochemical consistency of generated molecules
        - Exporting results to CSV (and optionally SDF)
        - Producing visualization of the optimal state

        Parameters
        ----------
        distribution
            Microstate distribution for the selected pH value.
        """

        if distribution.state_qs is None:
            raise ValueError("Microstate distribution is missing state charges.")

        # Max freq
        state_freq_max = np.max(distribution.state_freqs)

        # Select states for pH-specific export
        state_strs_export: list[str] = []
        state_freqs_export: list[float] = []

        for state_str, state_freq in zip(distribution.state_strs, distribution.state_freqs):
            if state_freq >= self.cutoff_export * state_freq_max:  # Include all high prob states
                state_strs_export.append(state_str)
                state_freqs_export.append(state_freq)

        state_freqs_arr: NDArray[np.float64] = np.array(state_freqs_export)
        ps = np.argsort(state_freqs_arr)[::-1]  # Sort by highest probability

        state_freqs_export = [state_freqs_export[p] for p in ps]
        state_strs_export = [state_strs_export[p] for p in ps]

        self.check_chiral_consistency(distribution.state_strs, distribution.indices)
        space = self.index_spaces.get(distribution.indices)

        logger.debug(f"Export at pH {self.pH}:")
        for e_idx, (state_str, sfreq) in enumerate(zip(state_strs_export, state_freqs_export)):
            logger.debug(e_idx, state_str, sfreq)

        molecule = combine_results(
            self.name,
            state_strs_export,
            state_freqs_export,
            space.mols_lib,
            distribution.state_qs,
        )
        return molecule

    def check_chiral_consistency(
        self,
        state_strs: list[str],
        indices: list[int],
    ) -> None:
        """Ensure generated microstate molecules can be embedded consistently.

        This method attempts to generate 3D embeddings for each microstate
        molecule. If embedding fails due to stereochemical constraints,
        chiral tags are removed and embedding is retried.

        Updated molecules are written back to the internal molecular cache.

        Note that this globally removes chirality when embedding fails!

        Parameters
        ----------
        state_strs
            Encoded microstate representations to validate.
        indices
            Absolute atom indices defining the current cluster.
        """

        space = self.index_spaces.get(indices)
        for state_str in state_strs:
            mol = space.mols_lib[state_str]

            mol_h = Chem.AddHs(mol)

            cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True)
            if cid != 0:
                logger.warning(f"Needed to remove chirality for embedding for {state_str}!")
                for atom in mol_h.GetAtoms():
                    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            cid = AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True)

            if cid != 0:
                raise ValueError(f"{state_str} could not be embedded.")
            for atom in mol.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

            space.mols_lib[state_str] = mol
