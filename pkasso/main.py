"""Core pKasso workflow implementation."""

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, RegistrationHash
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Mol

from . import coupling, special_cases, utils
from .predict_pka import (
    ModelInput,
    MolgpkaPredictor,
    Predictor,
    ResolvedPredictor,
    ThermodynamicPredictionMode,
    resolve_models,
)
from .external.molgpka.pka import configure_torch_threads
from .postprocess import Molecule, Scan, combine_results
from .transitions import (
    calc_freqs_from_states,
    calc_populations,
    calc_state_diffs,
    calc_state_pH_dependent_free_energies,
)
from .utils import pack_indices, pack_vec, unpack_vec, construct_mol
from .tautomers import best_tautomer_smiles

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.debug")
RDLogger.DisableLog("rdApp.info")


def _suppress_standardizer_debug_logs() -> None:
    """Hide routine standardization messages from verbose application loggers."""

    # Libraries such as OpenFF NAGL may globally re-enable RDKit's native log
    # channels. Reset the noisy channels on every call while preserving genuine
    # warnings and errors.
    RDLogger.DisableLog("rdApp.debug")
    RDLogger.DisableLog("rdApp.info")

    for namespace in ("rdkit.Chem.MolStandardize", "molvs"):
        logging.getLogger(namespace).setLevel(logging.INFO)
        for module in ("fragment", "normalize", "charge"):
            logging.getLogger(f"{namespace}.{module}").setLevel(logging.INFO)


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
    score_window: int = 0,
    num_threads: int = 1,
    min_fragment_heavy_atoms: int = 6
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

    # Apply this for every call because notebooks may reconfigure logging after import.
    _suppress_standardizer_debug_logs()

    logger.debug("Raw:")
    logger.debug(smiles_raw)
    mol = Chem.MolFromSmiles(smiles_raw, sanitize=True)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles_raw}")
    
    if strip_fragments:
        sizeable = sizeable_organic_fragments(mol,min_heavy_atoms=min_fragment_heavy_atoms)
        if len(sizeable) > 1:
            logger.warning(f"Input SMILES contains multiple sizeable organic fragments: {sizeable}")

        # Remove ions and covalent fragments
        chooser = rdMolStandardize.LargestFragmentChooser()
        mol = chooser.choose(mol)

    smiles = Chem.MolToSmiles(mol, canonical=True)

    logger.debug("Canonical")
    logger.debug(smiles)

    logger.debug("Formal charges before cleanup")
    charges = [at.GetFormalCharge() for at in mol.GetAtoms()]
    logger.debug(charges)

    mol = rdMolStandardize.Normalize(mol)
    uncharger = rdMolStandardize.Uncharger(force=True)
    mol = uncharger.uncharge(mol)
    
    # load/save cycles to clean up the mol atom ordering
    smiles = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    smiles = Chem.MolToSmiles(mol, canonical=True)

    if tautomer_search:
        smiles = best_tautomer_smiles(
            smiles,
            max_tautomers=max_tautomers,
            num_confs=num_confs,
            score_window=score_window,
            num_threads=num_threads,
        )
    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    logger.debug("Formal charges before cleanup")
    charges = [at.GetFormalCharge() for at in mol.GetAtoms()]
    logger.debug(charges)

    mol = rdMolStandardize.Normalize(mol)
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

    return mol, smiles


def _can_embed_with_stereochemistry(mol: Mol) -> bool:
    """Return whether RDKit can embed a sanitized copy of ``mol``."""

    mol_h = Chem.AddHs(Chem.Mol(mol))
    with rdBase.BlockLogs():
        return bool(AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True) == 0)


def _validate_smiles_roundtrip_and_embedding(mol: Mol) -> None:
    """Ensure the final isomeric SMILES can be loaded and embedded."""

    output_mol = Chem.Mol(mol)
    for atom in output_mol.GetAtoms():
        atom.SetAtomMapNum(0)

    smiles = Chem.MolToSmiles(output_mol, isomericSmiles=True)
    roundtrip = Chem.MolFromSmiles(smiles, sanitize=True)
    if roundtrip is None:
        raise ValueError(f"Generated microstate SMILES could not be loaded: {smiles}")
    if not _can_embed_with_stereochemistry(roundtrip):
        raise ValueError(f"Generated microstate SMILES could not be embedded: {smiles}")


def relax_stereochemistry_for_embedding(
    mol: Mol,
    changed_map_indices: set[int] | None = None,
) -> Mol:
    """Return an embeddable copy while retaining compatible atom stereochemistry.

    If the fully specified molecule cannot be embedded, all atom chiral tags are
    removed on a trial copy and then restored one at a time. Unchanged carbon
    stereocentres are restored first, while changed sites and heteroatom centres
    are restored later. This preserves the most stable stereochemical constraints
    without ever mutating the input molecule during failed embedding trials.
    """

    changed_map_indices = changed_map_indices or set()
    original = Chem.Mol(mol)
    Chem.SanitizeMol(original)

    if _can_embed_with_stereochemistry(original):
        _validate_smiles_roundtrip_and_embedding(original)
        return original

    chiral_atoms = [
        (atom.GetIdx(), atom.GetAtomMapNum(), atom.GetAtomicNum(), atom.GetChiralTag())
        for atom in original.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]

    relaxed = Chem.Mol(original)
    for atom in relaxed.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    if not _can_embed_with_stereochemistry(relaxed):
        raise ValueError(
            "Microstate could not be embedded even after removing all atom stereochemistry."
        )

    def restore_priority(item: tuple[int, int, int, Chem.ChiralType]) -> tuple[bool, bool, int]:
        atom_idx, map_idx, atomic_num, _ = item
        return atomic_num != 6, map_idx in changed_map_indices, atom_idx

    removed_centres: list[int] = []
    for atom_idx, map_idx, _, chiral_tag in sorted(chiral_atoms, key=restore_priority):
        trial = Chem.Mol(relaxed)
        trial.GetAtomWithIdx(atom_idx).SetChiralTag(chiral_tag)
        if _can_embed_with_stereochemistry(trial):
            relaxed = trial
        else:
            removed_centres.append(map_idx or atom_idx + 1)

    if removed_centres:
        logger.warning(
            "Removed incompatible atom stereochemistry at atom map indices %s.",
            removed_centres,
        )

    _validate_smiles_roundtrip_and_embedding(relaxed)
    return relaxed


def find_candidate_sites(
    base_map_ids: list[int],
    acid_map_ids: list[int],
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
        Atom base map ids.
    acid
        Atom acid map ids.
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

    indices_raw = list(sorted(set(base_map_ids + acid_map_ids)))
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
        if map_idx in base_map_ids:
            if map_idx not in exclude_base_indices:
                q_options[rel_idx, 2] = 1  # allow protonation
        if map_idx in acid_map_ids:
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


def _coerce_standard_free_energy_values(result: Any, expected_count: int) -> list[float]:
    """Extract ordered standard free energies from a predictor result."""

    if hasattr(result, "columns") and "standard_free_energy" in result.columns:
        values = result["standard_free_energy"].tolist()
    elif isinstance(result, dict) and "standard_free_energy" in result:
        values = list(result["standard_free_energy"])
    else:
        values = list(result)

    if len(values) != expected_count:
        raise ValueError(f"Expected {expected_count} standard free energies, got {len(values)}.")
    return [float(value) for value in values]

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
    standard_free_energy_lib: dict[str, float] = field(default_factory=dict)

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
class RawMicrostateEnergies:
    """Raw pH-dependent microstate free energies before population analysis."""

    index_space: ProtonationIndexSpace
    pH: float
    state_strs: list[str]
    state_vecs: list[NDArray[np.int64]]
    Gs: list[float] | NDArray[np.float64]
    expert_state_freqs: NDArray[np.float64] | None = None

    @property
    def indices(self) -> list[int]:
        return self.index_space.indices

    @property
    def mols_lib(self) -> dict[str, Mol]:
        return self.index_space.mols_lib


@dataclass
class PredictorContext:
    """Model-specific setup state and prediction caches."""

    predictor_cls: type[Predictor]
    predictor_config: object | None = None
    index_spaces: IndexSpaceRegistry = field(default_factory=IndexSpaceRegistry)
    exclude_base_indices: list[int] = field(default_factory=list)
    exclude_acid_indices: list[int] = field(default_factory=list)
    acid_map_ids: list[int] = field(default_factory=list)
    base_map_ids: list[int] = field(default_factory=list)
    acid0: dict[int, float] = field(default_factory=dict)
    base0: dict[int, float] = field(default_factory=dict)
    indices0: list[int] = field(default_factory=list)
    q_options0: NDArray[np.int64] | None = None
    index_space0: ProtonationIndexSpace | None = None
    clusters: list[list[int]] = field(default_factory=list)
    cluster_spaces: list[ProtonationIndexSpace] = field(default_factory=list)
    dist_raw: RawMicrostateEnergies | None = None


@dataclass
class MicrostateDistribution:
    """Final pH-dependent microstate distribution over one fixed index space."""

    index_space: ProtonationIndexSpace
    pH: float
    state_strs: list[str]
    state_vecs: list[NDArray[np.int64]]
    Gs: list[float] | NDArray[np.float64]
    state_freqs: list[float] | NDArray[np.float64]
    state_freqs_sigmas: list[float] | NDArray[np.float64] | None = None
    state_freq_samples: NDArray[np.float64] | None = None
    state_qs: dict[str, int] | None = None
    net_charge: float | None = None
    net_charge_sigma: float | None = None
    freqs_macro: dict[int, float] | None = None
    freqs_macro_sigmas: dict[int, float] | None = None
    freqs_macro_samples: dict[int, NDArray[np.float64]] | None = None

    @property
    def indices(self) -> list[int]:
        return self.index_space.indices

    @property
    def mols_lib(self) -> dict[str, Mol]:
        return self.index_space.mols_lib

    def apply_symmetry(self) -> None:
        """Merge symmetry-equivalent states and keep state fields aligned."""

        state_hashes = calc_hashes(self.state_strs, self.mols_lib)
        groups: dict[str, list[int]] = {}
        for state_idx, state_hash in enumerate(state_hashes):
            groups.setdefault(state_hash, []).append(state_idx)

        state_strs_symm: list[str] = []
        state_freqs_symm: list[float] = []
        state_freq_samples_symm: list[NDArray[np.float64]] | None = [] if self.state_freq_samples is not None else None

        for group in groups.values():
            state_strs_group = sorted(self.state_strs[state_idx] for state_idx in group)
            state_strs_symm.append(state_strs_group[0])
            state_freqs_symm.append(float(np.sum([self.state_freqs[state_idx] for state_idx in group])))
            if state_freq_samples_symm is not None and self.state_freq_samples is not None:
                state_freq_samples_symm.append(np.sum(self.state_freq_samples[:, group], axis=1))

        self.state_strs = state_strs_symm
        self.state_freqs = state_freqs_symm
        self.state_freqs = np.asarray(self.state_freqs, dtype=np.float64)
        self.Gs = -np.log(self.state_freqs)
        self.Gs -= np.min(self.Gs)
        if state_freq_samples_symm is not None:
            self.state_freq_samples = np.asarray(state_freq_samples_symm, dtype=np.float64).T
            self.state_freqs_sigmas = np.std(self.state_freq_samples, axis=0, ddof=1)
        self.state_vecs = [unpack_vec(state_str) for state_str in self.state_strs]

    def assign_macro_props(self) -> None:
        """Compute and store charge-resolved macrostate properties."""

        self.state_qs = calc_state_qs(self.state_strs, self.mols_lib)
        self.net_charge, self.freqs_macro = calc_macro_props(
            self.state_strs,
            self.state_freqs,
            self.state_qs,
        )
        if self.state_freq_samples is None:
            return

        freqs_macro_samples: dict[int, NDArray[np.float64]] = {}
        for state_idx, state_str in enumerate(self.state_strs):
            state_q = self.state_qs[state_str]
            if state_q not in freqs_macro_samples:
                freqs_macro_samples[state_q] = np.zeros(self.state_freq_samples.shape[0], dtype=np.float64)
            freqs_macro_samples[state_q] += self.state_freq_samples[:, state_idx]

        self.freqs_macro_samples = freqs_macro_samples
        self.freqs_macro_sigmas = {
            q: float(np.std(freq_samples, ddof=1))
            for q, freq_samples in freqs_macro_samples.items()
        }
        net_charge_samples = np.zeros(self.state_freq_samples.shape[0], dtype=np.float64)
        for q, freq_samples in freqs_macro_samples.items():
            net_charge_samples += q * freq_samples
        self.net_charge_sigma = float(np.std(net_charge_samples, ddof=1))


@dataclass
class PHScanDistribution:
    """Internal raw pH-scan distribution before public Scan postprocessing.

    This is the scan-level counterpart to MicrostateDistribution: it stores
    numerical pH-scan data produced by the core engine. The public Scan class
    in postprocess.py remains the outward-facing result with plotting and
    export helpers. Finalized per-pH distributions are retained so molecule
    output can be materialized on demand.
    """

    pHs: NDArray[np.float64]
    net_charges: list[float]
    net_charge_sigmas: list[float]
    state_freqs_all: dict[str, NDArray[np.float64]]
    state_freqs_sigmas_all: dict[str, NDArray[np.float64]]
    state_freqs_expert_max_all: dict[str, NDArray[np.float64]]
    freqs_macro_all: list[dict[int, float]]
    microstate_distributions: dict[float, MicrostateDistribution]
    freqs_macro_samples_all: list[dict[int, NDArray[np.float64]]] = field(default_factory=list)


def combine_cluster_distributions(
    cluster_dists: list[RawMicrostateEnergies],
    index_space: ProtonationIndexSpace,
    pH: float,
    free_energy_cutoff_combined: float = 30.0,
    max_states_combined: int = 100,
) -> RawMicrostateEnergies:
    """
    Combine microstate free energies from independent pKa clusters.

    Microstates from different clusters are combined assuming statistical
    independence, with the combined microstate free energy calculated as:

        G(AB) = G(A) + G(B)

    After generating all possible combinations, high-energy states are
    filtered relative to the lowest-energy combination.

    Parameters
    ----------
    cluster_dists
        pH-dependent microstate free energies for independent clusters.
    index_space
        Combined protonation state space that the output distribution belongs to.
    pH
        Current pH value.
    free_energy_cutoff_combined
        Maximum relative free energy for a combined microstate to be kept.
    max_states_combined
        Max. number of microstates at given pH value

    Returns
    -------
    microstate_energies
        Combined raw microstate free energies over ``index_space``.
    """

    state_strs_clusters = [dist.state_strs for dist in cluster_dists]
    Gs_clusters = [np.asarray(dist.Gs, dtype=np.float64) for dist in cluster_dists]
    indices_clusters = [dist.indices for dist in cluster_dists]

    logger.debug(state_strs_clusters)
    logger.debug(Gs_clusters)

    cluster_state_ids = [range(len(Gs_cl)) for Gs_cl in Gs_clusters]

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

    Gs_min = float(np.sum([np.min(Gs_cl) for Gs_cl in Gs_clusters]))
    G_cutoff = Gs_min + free_energy_cutoff_combined
    sort_state_str = not np.array_equal(ps, np.arange(len(ps)))

    state_strs = []
    Gs_list = []

    combinations = itertools.product(*cluster_state_ids)
    for s_idxs in combinations:
        G_comb = 0.0
        state_str_parts = []
        for c_idx, s_idx in enumerate(s_idxs):
            state_str_parts.append(state_strs_clusters[c_idx][s_idx])
            G_comb += float(Gs_clusters[c_idx][s_idx])

        if G_comb <= G_cutoff:
            state_str = "".join(state_str_parts)
            if sort_state_str:
                state_str = utils.sort_string(state_str, ps)  # match sorted indices
            state_strs.append(state_str)
            Gs_list.append(G_comb)

    Gs = np.array(Gs_list)

    if len(state_strs) > max_states_combined:
        ps_state_strs = np.argsort(Gs)
        state_strs = [state_strs[p] for p in ps_state_strs][:max_states_combined]
        Gs = Gs[ps_state_strs][:max_states_combined]

    logger.debug(f"N chosen microstate combinations: {len(state_strs)}")
    return RawMicrostateEnergies(
        index_space=index_space,
        pH=pH,
        state_strs=state_strs,
        state_vecs=[unpack_vec(state_str) for state_str in state_strs],
        Gs=Gs,
    )


def _normalized_weights(
    weights: Sequence[float] | None,
    n_weights: int,
) -> NDArray[np.float64]:
    if weights is None:
        return np.full(n_weights, 1.0 / n_weights, dtype=np.float64)

    weights_arr = np.asarray(weights, dtype=np.float64)
    if weights_arr.shape != (n_weights,):
        raise ValueError(f"Expected {n_weights} expert weights, got {len(weights_arr)}.")
    if np.any(weights_arr <= 0.0):
        raise ValueError("Expert weights must be positive.")
    weight_sum = float(np.sum(weights_arr))
    return weights_arr / weight_sum


def _state_population_map(G_by_state: dict[str, float]) -> dict[str, float]:
    """Compute Boltzmann populations over one expert's finite state set."""

    finite_items = [
        (state_str, G)
        for state_str, G in G_by_state.items()
        if np.isfinite(G)
    ]
    if not finite_items:
        return {}

    state_strs = [state_str for state_str, _ in finite_items]
    Gs = np.array([G for _, G in finite_items], dtype=np.float64)
    Gs -= np.min(Gs)
    pops = calc_populations(Gs)
    return dict(zip(state_strs, map(float, pops)))


def _align_energy_rows(
    G_by_state_rows: list[dict[str, float]],
    shared_states: set[str],
) -> list[dict[str, float]]:
    """Align model free-energy ladders to the first model.

    The offset is fitted over shared states, weighted by the geometric mean of
    each model's population over its full finite state set. This keeps the
    gauge alignment anchored to thermodynamically relevant shared states.
    """

    reference_row = G_by_state_rows[0]
    population_rows = [_state_population_map(G_by_state) for G_by_state in G_by_state_rows]
    aligned_rows: list[dict[str, float]] = []
    for row_idx, G_by_state in enumerate(G_by_state_rows):
        if row_idx == 0:
            offset = 0.0
        else:
            diffs = np.array([
                G_by_state[state_str] - reference_row[state_str]
                for state_str in shared_states
            ], dtype=np.float64)
            weights = np.array([
                np.sqrt(
                    population_rows[0].get(state_str, 0.0)
                    * population_rows[row_idx].get(state_str, 0.0)
                )
                for state_str in shared_states
            ], dtype=np.float64)
            if np.sum(weights) > 0.0:
                offset = float(np.average(diffs, weights=weights))
            else:
                offset = float(np.mean(diffs))
        aligned_rows.append({
            state_str: G - offset
            for state_str, G in G_by_state.items()
            if np.isfinite(G)
        })
    return aligned_rows


def _state_supported_by_expert(
    state_str: str,
    combined_indices: list[int],
    expert_space: ProtonationIndexSpace,
) -> bool:
    """Return whether an expert's original state space can represent a state."""

    options_by_index = {
        map_idx: expert_space.q_options[rel_idx]
        for rel_idx, map_idx in enumerate(expert_space.indices)
    }
    for map_idx, state_char in zip(combined_indices, state_str):
        state_value = int(state_char)
        options = options_by_index.get(map_idx)
        if options is None:
            if state_value != 1:
                return False
        elif options[state_value] == 0:
            return False
    return True


def _expert_population_rows(
    state_strs: list[str],
    G_by_state_rows: list[dict[str, float]],
    raw_dists: Sequence[RawMicrostateEnergies],
    combined_indices: list[int],
    pH: float,
) -> NDArray[np.float64] | None:
    """Return expert populations, treating supported pruned states as zero."""

    population_rows = []
    state_indices = {
        state_str: state_idx
        for state_idx, state_str in enumerate(state_strs)
    }
    for expert_idx, (G_by_state, raw) in enumerate(zip(G_by_state_rows, raw_dists)):
        missing_states = [
            state_str
            for state_str in state_strs
            if state_str not in G_by_state
        ]
        unsupported_states = [
            state_str
            for state_str in missing_states
            if not _state_supported_by_expert(
                state_str,
                combined_indices,
                raw.index_space,
            )
        ]
        if unsupported_states:
            for state_str in unsupported_states:
                logger.warning(
                    "Microstate %s at pH %.3f is unsupported by expert %d. "
                    "Population uncertainty is unavailable.",
                    state_str,
                    pH,
                    expert_idx + 1,
                )
            return None

        evaluated_states = [
            state_str
            for state_str in state_strs
            if state_str in G_by_state
        ]
        if not evaluated_states:
            logger.warning(
                "Expert %d retained no states from the authoritative state space "
                "at pH %.3f. Population uncertainty is unavailable.",
                expert_idx + 1,
                pH,
            )
            return None

        if missing_states:
            logger.debug(
                "Treating supported states %s pruned by expert %d at pH %.3f "
                "as zero-population states.",
                missing_states,
                expert_idx + 1,
                pH,
            )

        evaluated_populations = calc_populations(np.asarray([
            G_by_state[state_str]
            for state_str in evaluated_states
        ]))
        populations = np.zeros(len(state_strs), dtype=np.float64)
        for state_str, population in zip(evaluated_states, evaluated_populations):
            populations[state_indices[state_str]] = population
        population_rows.append(populations)

    return np.asarray(population_rows, dtype=np.float64)


def combine_expert_energies(
    raw_dists: Sequence[RawMicrostateEnergies],
    *,
    index_space: ProtonationIndexSpace | None = None,
    method: str = "product_of_experts",
    weights: Sequence[float] | None = None,
) -> RawMicrostateEnergies:
    """Combine aligned raw free-energy predictions from multiple experts.

    The first expert defines the admissible post-pruning state set. Later
    experts contribute only for states also retained by that reference expert;
    their exclusive states are ignored.
    """

    if not raw_dists:
        raise ValueError("At least one raw distribution is required.")
    if len(raw_dists) == 1:
        raw = raw_dists[0]
        if index_space is None or index_space is raw.index_space:
            return raw
        return RawMicrostateEnergies(
            index_space=index_space,
            pH=raw.pH,
            state_strs=list(raw.state_strs),
            state_vecs=list(raw.state_vecs),
            Gs=np.asarray(raw.Gs, dtype=np.float64),
        )

    reference = raw_dists[0]
    pH = reference.pH
    combined_indices = sorted({
        idx
        for raw in raw_dists
        for idx in raw.index_space.indices
    })
    combined_q_options = np.zeros((len(combined_indices), 3), dtype=np.int64)
    combined_q_options[:, 1] = 1
    for raw in raw_dists:
        for rel_idx, map_idx in enumerate(raw.index_space.indices):
            combined_rel_idx = combined_indices.index(map_idx)
            combined_q_options[combined_rel_idx] = np.maximum(
                combined_q_options[combined_rel_idx],
                raw.index_space.q_options[rel_idx],
            )

    if index_space is not None and index_space.indices == combined_indices:
        combined_index_space = index_space
    elif (
        reference.index_space.indices == combined_indices
        and np.array_equal(reference.index_space.q_options, combined_q_options)
    ):
        combined_index_space = reference.index_space
    else:
        combined_index_space = ProtonationIndexSpace(
            indices=combined_indices,
            q_options=combined_q_options,
        )

    G_by_state_rows: list[dict[str, float]] = []
    state_sets: list[set[str]] = []
    for raw in raw_dists:
        if not np.isclose(raw.pH, pH):
            raise ValueError("Expert raw distributions must use the same pH.")

        state_map = {
            state_str: "".join(
                state_str[raw.index_space.indices.index(map_idx)]
                if map_idx in raw.index_space.indices
                else "1"
                for map_idx in combined_indices
            )
            for state_str in raw.state_strs
        }
        G_by_state = {
            state_map[state_str]: float(G) for state_str, G in zip(raw.state_strs, raw.Gs)
        }
        finite_states = {state_str for state_str, G in G_by_state.items() if np.isfinite(G)}
        if not finite_states:
            raise ValueError("Expert raw distribution contains no finite free energies.")
        G_by_state_rows.append(G_by_state)
        state_sets.append(finite_states)

    reference_states = state_sets[0]
    for row_idx in range(1, len(G_by_state_rows)):
        G_by_state_rows[row_idx] = {
            state_str: G
            for state_str, G in G_by_state_rows[row_idx].items()
            if state_str in reference_states
        }
        state_sets[row_idx] = set(G_by_state_rows[row_idx])

    shared_states = set.intersection(*state_sets)
    if not shared_states:
        logger.info(
            "Expert raw distributions do not share any finite microstate strings. "
            "Falling back to the first model provided."
        )
        fallback_G_by_state = G_by_state_rows[0]
        fallback_state_strs = [
            state_str
            for state_str in fallback_G_by_state
            if state_str in state_sets[0]
        ]
        expert_state_freqs = _expert_population_rows(
            fallback_state_strs,
            G_by_state_rows,
            raw_dists,
            combined_indices,
            pH,
        )
        return RawMicrostateEnergies(
            index_space=combined_index_space,
            pH=pH,
            state_strs=fallback_state_strs,
            state_vecs=[unpack_vec(state_str) for state_str in fallback_state_strs],
            Gs=np.array([fallback_G_by_state[state_str] for state_str in fallback_state_strs], dtype=np.float64),
            expert_state_freqs=expert_state_freqs,
        )

    aligned_rows = _align_energy_rows(G_by_state_rows, shared_states)

    combined_states = sorted(reference_states)
    state_vec_by_str = {
        state_str: unpack_vec(state_str)
        for state_str in combined_states
    }
    expert_state_freqs = _expert_population_rows(
        combined_states,
        aligned_rows,
        raw_dists,
        combined_indices,
        pH,
    )

    weights_arr = _normalized_weights(weights, len(raw_dists))

    if method == "product_of_experts":
        Gs_list = []
        for state_str in combined_states:
            available = [
                row_idx for row_idx, G_by_state in enumerate(aligned_rows)
                if state_str in G_by_state
            ]
            available_weights = weights_arr[available]
            available_weights = available_weights / np.sum(available_weights)
            Gs_list.append(float(np.sum([
                available_weights[idx] * aligned_rows[row_idx][state_str]
                for idx, row_idx in enumerate(available)
            ])))
        Gs = np.array(Gs_list, dtype=np.float64)
    elif method == "mixture_of_experts":
        pops_rows = []
        for G_by_state in aligned_rows:
            state_strs = list(G_by_state)
            pops = calc_populations(np.array([G_by_state[state_str] for state_str in state_strs], dtype=np.float64))
            pops_rows.append(dict(zip(state_strs, pops)))
        pops_matrix = np.array([
            [pops_by_state.get(state_str, 0.0) for state_str in combined_states]
            for pops_by_state in pops_rows
        ], dtype=np.float64)
        mixed_pops = np.sum(weights_arr[:, None] * pops_matrix, axis=0)
        positive = mixed_pops > 0.0
        if not np.any(positive):
            raise ValueError("Mixture of experts produced no positive populations.")
        Gs = np.full_like(mixed_pops, np.inf, dtype=np.float64)
        Gs[positive] = -np.log(mixed_pops[positive])
    else:
        raise ValueError("expert_combination must be 'product_of_experts' or 'mixture_of_experts'.")

    finite = np.isfinite(Gs)
    if not np.any(finite):
        raise ValueError("Combined expert distribution contains no finite free energies.")
    Gs -= np.min(Gs[finite])

    return RawMicrostateEnergies(
        index_space=combined_index_space,
        pH=pH,
        state_strs=combined_states,
        state_vecs=[state_vec_by_str[state_str] for state_str in combined_states],
        Gs=Gs,
        expert_state_freqs=expert_state_freqs,
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
                if freq1 <= 0.0 or freq2 <= 0.0:
                    continue
                pka_macro = np.log10(freq2 / freq1) + pH
                pka_weight = (freq1 * freq2) / (freq1 + freq2)
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


def combine_pkas_macro_sigmas(
    pHs: NDArray[np.float64],
    freqs_macro_samples_all: list[dict[int, NDArray[np.float64]]],
) -> dict[int, float]:
    """Estimate macro-pKa uncertainty from sampled macrostate frequencies."""

    if not freqs_macro_samples_all or any(
        not freqs_macro_samples
        for freqs_macro_samples in freqs_macro_samples_all
    ):
        return {}

    sample_counts = {
        len(next(iter(freqs_macro_samples.values())))
        for freqs_macro_samples in freqs_macro_samples_all
    }
    if len(sample_counts) != 1:
        return {}
    n_samples = sample_counts.pop()
    if n_samples < 2:
        return {}

    pkas_by_expert: list[dict[int, float]] = []
    for sample_idx in range(n_samples):
        freqs_macro_expert = [
            {
                q: float(freq_samples[sample_idx])
                for q, freq_samples in freqs_macro_samples.items()
            }
            for freqs_macro_samples in freqs_macro_samples_all
        ]
        pkas_by_expert.append(combine_pkas_macro(pHs, freqs_macro_expert))

    shared_qs = set.intersection(*[set(pkas) for pkas in pkas_by_expert])

    return {
        q: float(np.std([pkas[q] for pkas in pkas_by_expert], ddof=1))
        for q in shared_qs
    }


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
            name, cutoff_states, model,
            free_energy_cutoff_individual, free_energy_cutoff_combined,
            expert_combination, expert_weights,
            matrix_def, cutoff_export, nthreads

        ``model`` is an ordered mapping from predictor names to their options,
        for example ``{"molgpka": {}, "unipka": {"gpu": True}}``.

    """

    smiles: str
    name: str = "molecule"

    # Internal options
    cutoff_states: int = 200
    free_energy_cutoff_individual: float = 10.
    max_states_individual: int = 20
    free_energy_cutoff_combined: float = 30.0
    max_states_combined: int = 20
    cutoff_export: float = 0.2
    matrix_def: str = "dG"
    model: ModelInput = field(default_factory=lambda: {"molgpka": {}})
    expert_combination: str = "product_of_experts"
    expert_weights: Sequence[float] | None = None
    tautomer_search: bool = True
    max_tautomers: int = 20
    num_confs: int = 10
    total_max_sites: int = 25
    max_cut_edges: int = 1
    strip_fragments: bool = True
    score_window: int = 0
    nthreads: int = 0
    fragment_warning_heavy_atoms: int = 6
    resolved_predictors: tuple[ResolvedPredictor, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        configure_torch_threads(self.nthreads)
        self.resolved_predictors = resolve_models(self.model, nthreads=self.nthreads)

    def model_classes(self) -> tuple[type[Predictor], ...]:
        """Return the configured model classes in evaluation order."""

        return tuple(resolved.predictor_cls for resolved in self.resolved_predictors)

    def primary_predictor_cls(self) -> type[Predictor]:
        """Return the model used for setup and the first raw-energy pass."""

        return self.model_classes()[0]

    def primary_predictor_config(self) -> object | None:
        """Return the resolved configuration for the primary model."""

        return self.resolved_predictors[0].config

    def pka_predictor(
        self,
        mol: Mol,
        context: PredictorContext | None = None,
        *,
        source_mol: Mol | None = None,
    ) -> Predictor:
        """Create the configured molecule-specific pKa predictor."""

        predictor_cls = context.predictor_cls if context is not None else self.primary_predictor_cls()
        predictor = predictor_cls(mol)
        if source_mol is not None:
            predictor.source_mol = source_mol
        return predictor

    def uses_standard_free_energies(self, context: PredictorContext | None = None) -> bool:
        """Return whether this run should use direct microstate free energies."""

        return self.thermodynamic_prediction_mode(context) == "standard_free_energy"

    def thermodynamic_prediction_mode(
        self,
        context: PredictorContext | None = None,
    ) -> ThermodynamicPredictionMode:
        """Return the thermodynamic model route selected for this run."""

        predictor_cls = context.predictor_cls if context is not None else self.primary_predictor_cls()
        mode = getattr(predictor_cls, "thermodynamic_prediction", None)
        if mode not in ("pka", "standard_free_energy"):
            raise ValueError(
                "Predictor classes must define thermodynamic_prediction as "
                "'pka' or 'standard_free_energy'."
            )
        return cast(ThermodynamicPredictionMode, mode)
    
    def opposite_charge_influence_mode(self, context: PredictorContext | None = None) -> bool:
        predictor_cls = context.predictor_cls if context is not None else self.primary_predictor_cls()
        mode = getattr(predictor_cls, "opposite_charge_influence", True)
        return mode

    def molgpka_prediction_mode(self, context: PredictorContext | None = None) -> bool:
        """Return whether the current predictor context uses MolGpKa."""

        predictor_cls = context.predictor_cls if context is not None else self.primary_predictor_cls()
        return issubclass(predictor_cls, MolgpkaPredictor)

    def phosphate_relative_clusters(self, indices0: list[int]) -> list[list[int]]:
        """Return phosphate OH groups as relative site indices in ``indices0``."""

        if not hasattr(self, "mol0"):
            return []

        _, phosphate_groups = special_cases.has_phosphate(self.mol0)
        phosphate_clusters: list[list[int]] = []

        for oh_map_ids in phosphate_groups.values():
            phosphate_rel_indices = sorted(
                {
                    indices0.index(oh_map_idx)
                    for oh_map_idx in oh_map_ids
                    if oh_map_idx in indices0
                }
            )
            if phosphate_rel_indices:
                phosphate_clusters.append(phosphate_rel_indices)

        return phosphate_clusters

    def decouple_phosphate_clusters(
        self,
        clusters: list[list[int]],
        phosphate_clusters: list[list[int]],
    ) -> list[list[int]]:
        """Extract phosphate OH sites from existing clusters into phosphate-only clusters."""

        if not phosphate_clusters:
            return clusters

        phosphate_rel_indices = {
            rel_idx
            for phosphate_cluster in phosphate_clusters
            for rel_idx in phosphate_cluster
        }

        decoupled_clusters = []
        for cluster in clusters:
            non_phosphate_cluster = [
                rel_idx
                for rel_idx in cluster
                if rel_idx not in phosphate_rel_indices
            ]
            if non_phosphate_cluster:
                decoupled_clusters.append(non_phosphate_cluster)

        decoupled_clusters.extend(phosphate_clusters)
        return sorted(decoupled_clusters, key=lambda cluster: cluster[0])

    def standard_free_energy_target_mean(
        self,
        context: PredictorContext | None = None,
    ) -> float:
        """Return the target-mean offset declared by a free-energy predictor."""

        predictor_cls = context.predictor_cls if context is not None else self.primary_predictor_cls()
        target_mean = predictor_cls.standard_free_energy_target_mean
        if target_mean is None:
            raise ValueError(
                f"{predictor_cls.__name__} must define standard_free_energy_target_mean."
            )
        return float(target_mean)

    def predict_standard_free_energy_values(
        self,
        mols: list[Mol],
        context: PredictorContext | None = None,
    ) -> list[float]:
        """Predict ordered standard free energies for a batch of microstate mols."""

        if not mols:
            return []

        if context is None:
            predictor_cls = self.primary_predictor_cls()
            predictor_config = self.primary_predictor_config()
        else:
            predictor_cls = context.predictor_cls
            predictor_config = context.predictor_config
        result = predictor_cls.predict_standard_free_energies(
            mols,
            config=predictor_config,
        )

        return _coerce_standard_free_energy_values(result, len(mols))

    def run_single(self, pH: float = 7.0) -> Molecule:
        """
        Run the full pKasso pipeline.

        1. Setup of models and preprocessing.
        2. pH-dependent state enumeration and thermodynamic evaluation.
        3. Final analysis, pKa aggregation, and visualization.
        """

        self.pH = pH
        self._setup()
        dist = self._calc_microstates(self.pH)
        molecule = self.prep_single_output(dist)
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

        self.initialize_paths_models_libs()
        self.mol0, self.smiles0 = preprocess(
            self.smiles,
            tautomer_search=self.tautomer_search,
            max_tautomers=self.max_tautomers,
            num_confs=self.num_confs,
            strip_fragments=self.strip_fragments,
            score_window=self.score_window,
            num_threads=self.nthreads,
            min_fragment_heavy_atoms=self.fragment_warning_heavy_atoms
        )

        self.charged_indices = special_cases.find_charged(self.mol0)

        logger.debug("Processed SMILES:")
        logger.debug(self.smiles0)

        self.predictor_contexts = [
            self._setup_predictor_context(resolved)
            for resolved in self.resolved_predictors
        ]
        self.primary_context = self.predictor_contexts[0]
        self._bind_combined_context_space()

    def _setup_predictor_context(self, resolved: ResolvedPredictor) -> PredictorContext:
        """Create model-specific site, cluster, and cache state."""

        predictor_cls = resolved.predictor_cls
        context = PredictorContext(
            predictor_cls=predictor_cls,
            predictor_config=resolved.config,
        )
        pka_predictor = self.pka_predictor(self.mol0, context)
        context.exclude_base_indices, context.exclude_acid_indices = pka_predictor.exclude_sites()

        logger.debug(f"Predictor: {predictor_cls.__name__}")
        logger.debug(f"Exclude base indices: {context.exclude_base_indices}")
        logger.debug(f"Exclude acid indices: {context.exclude_acid_indices}")

        context.acid_map_ids = pka_predictor.pred_acid_ids()
        context.base_map_ids = pka_predictor.pred_base_ids()
        context.acid0 = pka_predictor.pred_acid()
        context.base0 = pka_predictor.pred_base()

        n_candidate_sites = len(set(context.acid_map_ids + context.base_map_ids))
        if n_candidate_sites > self.total_max_sites:
            logger.warning(
                f"Molecule has >{self.total_max_sites} protonation sites. "
                "Returning processed input molecule."
            )
            context.indices0 = []
            context.q_options0 = np.empty((0, 3), dtype=np.int64)
        else:
            context.indices0, context.q_options0 = find_candidate_sites(
                context.base_map_ids,
                context.acid_map_ids,
                context.exclude_base_indices,
                context.exclude_acid_indices,
                self.charged_indices,
            )
        context.index_space0 = context.index_spaces.get_or_create(
            context.indices0,
            context.q_options0,
        )
        context.clusters = self.screen_clusters(
            context.indices0,
            context.q_options0,
            context=context,
        )
        context.cluster_spaces = [
            context.index_spaces.get_or_create(
                [context.indices0[c] for c in cluster],
                context.q_options0[cluster],
            )
            for cluster in context.clusters
        ]
        logger.debug(f"Clusters: {context.clusters}")
        return context

    def _bind_combined_context_space(self) -> None:
        """Expose the pH-independent union space through legacy attributes."""

        combined_indices = sorted({
            idx
            for context in self.predictor_contexts
            for idx in context.indices0
        })
        combined_q_options = np.zeros((len(combined_indices), 3), dtype=np.int64)
        combined_q_options[:, 1] = 1
        for context in self.predictor_contexts:
            if context.q_options0 is None:
                raise ValueError("Predictor context is incomplete.")
            for rel_idx, map_idx in enumerate(context.indices0):
                combined_rel_idx = combined_indices.index(map_idx)
                combined_q_options[combined_rel_idx] = np.maximum(
                    combined_q_options[combined_rel_idx],
                    context.q_options0[rel_idx],
                )

        self.index_spaces = IndexSpaceRegistry()
        self.indices0 = combined_indices
        self.q_options0 = combined_q_options
        self.indices0_str = pack_indices(self.indices0)
        self.index_space0 = self.index_spaces.get_or_create(self.indices0, self.q_options0)

        self.exclude_base_indices = self.primary_context.exclude_base_indices
        self.exclude_acid_indices = self.primary_context.exclude_acid_indices
        self.acid_map_ids = self.primary_context.acid_map_ids
        self.base_map_ids = self.primary_context.base_map_ids
        self.acid0 = self.primary_context.acid0
        self.base0 = self.primary_context.base0
        self.clusters = self.primary_context.clusters
        self.cluster_spaces = self.primary_context.cluster_spaces

    def _calc_microstates_raw_for_context(
        self,
        context: PredictorContext,
        pH: float,
    ) -> RawMicrostateEnergies:
        """Calculate raw microstate free energies for one predictor context."""

        # indices0_curated, q_options0 = self.calc_curated_indices(pH)

        cluster_dists: list[RawMicrostateEnergies] = []

        for cluster_space in context.cluster_spaces:
            cluster_dists.append(
                self.process_cluster(
                    cluster_space,
                    pH,
                    free_energy_cutoff_individual=self.free_energy_cutoff_individual,
                    max_states_individual=self.max_states_individual,
                    context=context,
                )
            )

        if context.index_space0 is None:
            raise ValueError("Predictor context is missing the full index space.")
        context.dist_raw = combine_cluster_distributions(
            cluster_dists,
            context.index_space0,
            pH,
            free_energy_cutoff_combined=self.free_energy_cutoff_combined,
            max_states_combined=self.max_states_combined,
        )
        return context.dist_raw

    def _calc_microstates_raw(self, pH: float) -> list[RawMicrostateEnergies]:
        """Calculate one raw microstate free-energy distribution per model."""

        return [
            self._calc_microstates_raw_for_context(context, pH)
            for context in self.predictor_contexts
        ]

    def _finalize_microstates(
        self,
        raw: RawMicrostateEnergies | Sequence[RawMicrostateEnergies],
    ) -> MicrostateDistribution:
        """Convert raw free energies to final populations and macro properties."""

        raw_dists = [raw] if isinstance(raw, RawMicrostateEnergies) else list(raw)
        if not raw_dists:
            raise ValueError("At least one raw distribution is required.")
        index_space = getattr(self, "index_space0", raw_dists[0].index_space)
        raw_combined = combine_expert_energies(
            raw_dists,
            index_space=index_space,
            method=self.expert_combination,
            weights=self.expert_weights,
        )

        Gs = np.asarray(raw_combined.Gs, dtype=np.float64)
        finite = np.isfinite(Gs)
        if not np.any(finite):
            raise ValueError("Raw microstate distribution contains no finite free energies.")
        Gs = Gs - np.min(Gs[finite])
        state_freqs = calc_populations(Gs)
        state_freq_samples = raw_combined.expert_state_freqs
        state_freqs_sigmas = (
            None
            if state_freq_samples is None
            else np.std(state_freq_samples, axis=0, ddof=1)
        )

        dist = MicrostateDistribution(
            index_space=raw_combined.index_space,
            pH=raw_combined.pH,
            state_strs=list(raw_combined.state_strs),
            state_vecs=list(raw_combined.state_vecs),
            Gs=Gs,
            state_freqs=state_freqs,
            state_freqs_sigmas=state_freqs_sigmas,
            state_freq_samples=state_freq_samples,
        )

        self.construct_mols(dist.index_space, dist.state_strs, dist.state_vecs)
        dist.apply_symmetry()
        dist.assign_macro_props()
        return dist

    def _calc_microstates(self, pH: float) -> MicrostateDistribution:
        """Calculate finalized microstate frequencies for one pH value."""

        return self._finalize_microstates(self._calc_microstates_raw(pH))

    def _scan_pH(self, pHs: NDArray[np.float64]) -> PHScanDistribution:
        """
        Perform the full pH-dependent microstate enumeration and analysis.

        For each pH value in the configured pH grid, this method:

        - Identifies curated candidate titration sites
        - Screens residue coupling and builds clusters of coupled sites
        - Runs the finalized single-pH microstate calculation
        - Collects macrostate properties (net charge and macro-pKa data)
        - Stores results for later visualization
        """

        net_charges: list[float] = []
        net_charge_sigmas: list[float] = []
        state_freqs_all: dict[str, NDArray[np.float64]] = {}
        state_freqs_sigmas_all: dict[str, NDArray[np.float64]] = {}
        state_freqs_expert_max_all: dict[str, NDArray[np.float64]] = {}
        freqs_macro_all: list[dict[int, float]] = []
        freqs_macro_samples_all: list[dict[int, NDArray[np.float64]]] = []
        microstate_distributions: dict[float, MicrostateDistribution] = {}

        for pH_idx, pH in enumerate(pHs.flat):
            distribution = self._calc_microstates(float(pH))
            microstate_distributions[float(pH)] = distribution

            if distribution.net_charge is None or distribution.freqs_macro is None:
                raise ValueError("Microstate distribution is missing macro properties.")
            net_charges.append(distribution.net_charge)
            net_charge_sigmas.append(
                np.nan if distribution.net_charge_sigma is None else distribution.net_charge_sigma
            )
            freqs_macro_all.append(distribution.freqs_macro)
            freqs_macro_samples_all.append(distribution.freqs_macro_samples or {})

            # Add to results for pH scan
            state_freqs_sigmas = (
                np.full(len(distribution.state_freqs), np.nan, dtype=np.float64)
                if distribution.state_freqs_sigmas is None
                else np.asarray(distribution.state_freqs_sigmas, dtype=np.float64)
            )
            state_freqs_expert_max = (
                np.asarray(distribution.state_freqs, dtype=np.float64)
                if distribution.state_freq_samples is None
                else np.max(distribution.state_freq_samples, axis=0)
            )
            for state_str, state_freq, state_freq_sigma, state_freq_expert_max in zip(
                distribution.state_strs,
                distribution.state_freqs,
                state_freqs_sigmas,
                state_freqs_expert_max,
            ):
                if state_str not in state_freqs_all:
                    state_freqs_all[state_str] = np.zeros(len(pHs))
                    state_freqs_sigmas_all[state_str] = np.zeros(len(pHs))
                    state_freqs_expert_max_all[state_str] = np.zeros(len(pHs))
                state_freqs_all[state_str][pH_idx] = state_freq
                state_freqs_sigmas_all[state_str][pH_idx] = state_freq_sigma
                state_freqs_expert_max_all[state_str][pH_idx] = state_freq_expert_max

        return PHScanDistribution(
            pHs=pHs,
            net_charges=net_charges,
            net_charge_sigmas=net_charge_sigmas,
            state_freqs_all=state_freqs_all,
            state_freqs_sigmas_all=state_freqs_sigmas_all,
            state_freqs_expert_max_all=state_freqs_expert_max_all,
            freqs_macro_all=freqs_macro_all,
            freqs_macro_samples_all=freqs_macro_samples_all,
            microstate_distributions=microstate_distributions,
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
        net_charge_sigmas = np.array(distribution.net_charge_sigmas, dtype=np.float64)

        pkas_macro = combine_pkas_macro(distribution.pHs, distribution.freqs_macro_all)
        pkas_macro_sigmas = combine_pkas_macro_sigmas(
            distribution.pHs,
            distribution.freqs_macro_samples_all,
        )

        state_strs_relevant: list[str] = []
        sfreqs_relevant: list[NDArray[np.float64]] = []
        sfreqs_relevant_sigmas: list[NDArray[np.float64]] = []
        mols_relevant: list[Mol] = []
        sfreqs_not_relevant: list[NDArray[np.float64]] = []
        sfreqs_not_relevant_sigmas: list[NDArray[np.float64]] = []

        if distribution.state_freqs_all:
            (
                state_strs_relevant,
                sfreqs_relevant,
                sfreqs_relevant_sigmas,
                mols_relevant,
                sfreqs_not_relevant,
                sfreqs_not_relevant_sigmas,
            ) = self.calc_relevant_states(
                distribution.state_freqs_all,
                distribution.state_freqs_sigmas_all,
                distribution.state_freqs_expert_max_all,
            )

        return Scan(
            self.name,
            self.indices0,
            state_strs_relevant,
            mols_relevant,
            sfreqs_relevant,
            sfreqs_relevant_sigmas,
            distribution.pHs,
            net_charges,
            net_charge_sigmas,
            sfreqs_not_relevant,
            sfreqs_not_relevant_sigmas,
            pkas_macro,
            pkas_macro_sigmas,
            microstate_distributions=distribution.microstate_distributions,
            _molecule_factory=self.prep_single_output,
        )

    def calc_relevant_states(
        self,
        state_freqs_all: dict[str, NDArray[np.float64]],
        state_freqs_sigmas_all: dict[str, NDArray[np.float64]] | None = None,
        state_freqs_expert_max_all: dict[str, NDArray[np.float64]] | None = None,
        max_states: int = 18,
    ) -> tuple[
        list[str],
        list[NDArray[np.float64]],
        list[NDArray[np.float64]],
        list[Mol],
        list[NDArray[np.float64]],
        list[NDArray[np.float64]],
    ]:
        """Reduce number of states to max_states for plotting."""

        cutoff = 0.01
        if state_freqs_sigmas_all is None:
            state_freqs_sigmas_all = {
                state_str: np.zeros_like(sfreqs, dtype=np.float64)
                for state_str, sfreqs in state_freqs_all.items()
            }
        if state_freqs_expert_max_all is None:
            state_freqs_expert_max_all = state_freqs_all

        while True:
            state_strs_relevant: list[str] = []
            sfreqs_relevant: list[NDArray[np.float64]] = []
            sfreqs_relevant_sigmas: list[NDArray[np.float64]] = []
            sfreqs_not_relevant: list[NDArray[np.float64]] = []
            sfreqs_not_relevant_sigmas: list[NDArray[np.float64]] = []
            mols_relevant: list[Mol] = []
            pH_argmaxs: list[int] = []

            for state_str, sfreqs in state_freqs_all.items():
                mol = Chem.Mol(self.index_space0.mols_lib[state_str])
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)

                if (
                    np.max(sfreqs) > cutoff
                    or np.max(state_freqs_expert_max_all[state_str]) > cutoff
                ):
                    state_strs_relevant.append(state_str)
                    sfreqs_relevant.append(sfreqs)
                    sfreqs_relevant_sigmas.append(state_freqs_sigmas_all[state_str])
                    mols_relevant.append(mol)
                    pH_argmaxs.append(int(np.argmax(sfreqs)))
                else:
                    sfreqs_not_relevant.append(sfreqs)
                    sfreqs_not_relevant_sigmas.append(state_freqs_sigmas_all[state_str])

            N_relevant_states = len(state_strs_relevant)
            if N_relevant_states <= max_states:
                break

            cutoff += 0.02

        ps: list[int] = [int(p) for p in np.argsort(pH_argmaxs)]
        mols_relevant = [mols_relevant[p] for p in ps]
        for idx, mol in enumerate(mols_relevant, start=1):
            q = Chem.GetFormalCharge(mol)
            qstr = f"{q:+d}"
            mol.SetProp("_Name", f"{idx} ({qstr})")

        logger.debug(f"Final N relevant states: {N_relevant_states} with cutoff {cutoff}")
        return (
            [state_strs_relevant[p] for p in ps],
            [sfreqs_relevant[p] for p in ps],
            [sfreqs_relevant_sigmas[p] for p in ps],
            mols_relevant,
            sfreqs_not_relevant,
            sfreqs_not_relevant_sigmas,
        )

    #########################

    def initialize_paths_models_libs(self) -> None:
        """
        Reset internal libraries used to cache state-dependent predictions.
        """

        self.index_spaces = IndexSpaceRegistry()
        self.predictor_contexts = []

    def process_cluster(
        self,
        space: ProtonationIndexSpace,
        pH: float,
        free_energy_cutoff_individual: float = 7.0,
        max_states_individual: int = 100,
        context: PredictorContext | None = None,

    ) -> RawMicrostateEnergies:
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
        microstate_energies
            pH-dependent raw microstate free energies for this cluster.
        """

        state_vecs = construct_state_vectors(space.q_options, self.cutoff_states)

        state_strs = utils.calc_state_strs(state_vecs)

        self.construct_mols(space, state_strs, state_vecs)

        if self.uses_standard_free_energies(context):
            self.run_standard_free_energy_calcs(space, state_strs, context)
            standard_free_energies = np.array(
                [space.standard_free_energy_lib[state_str] for state_str in state_strs],
                dtype=np.float64,
            )
            Gs = calc_state_pH_dependent_free_energies(
                standard_free_energies,
                state_vecs,
                pH,
                self.standard_free_energy_target_mean(context),
            )
            Gs -= np.min(Gs)
        else:
            self.run_acid_base_calcs(space, state_strs, state_vecs, context)
            
            ps_all = calc_state_diffs(
                state_strs,
                state_vecs,
                space.indices,
                space.base_lib,
                space.acid_lib,
                pH=pH,
                matrix_def=self.matrix_def,
            )
            
            state_strs, Gs = calc_freqs_from_states(
                state_strs,
                state_vecs,
                ps_all,
                self.matrix_def,
            )

        # Cull

        ps = np.argsort(Gs)
        state_strs = [state_strs[p] for p in ps][:max_states_individual]
        Gs = Gs[ps][:max_states_individual]

        min_G = np.min(Gs)

        state_strs_list = []
        Gs_list = []

        for state_str, G in zip(state_strs, Gs):
            if (G - min_G) <= free_energy_cutoff_individual:
                state_strs_list.append(state_str)
                Gs_list.append(G)

        state_strs = state_strs_list
        Gs = np.array(Gs_list)

        state_vecs = [unpack_vec(state_str) for state_str in state_strs]
        return RawMicrostateEnergies(
            index_space=space,
            pH=pH,
            state_strs=state_strs,
            state_vecs=state_vecs,
            Gs=Gs,
        )

    #########################

    def coupling_assay_weights(
        self,
        indices: list[int],
        q_options: NDArray[np.int64],
        standard_free_energy_lib: dict[str, float] | None = None,
        context: PredictorContext | None = None,
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

        index_spaces = context.index_spaces if context is not None else self.index_spaces
        space = index_spaces.get_or_create(indices, q_options)

        if standard_free_energy_lib is not None:
            space.standard_free_energy_lib.update(standard_free_energy_lib)

        if self.uses_standard_free_energies(context) or space.standard_free_energy_lib:
            state_vecs_screen = coupling.construct_state_vectors_coupling(indices, q_options)
            state_strs_screen = utils.calc_state_strs(state_vecs_screen)
            self.construct_mols(space, state_strs_screen, state_vecs_screen)

            if self.uses_standard_free_energies(context):
                self.run_standard_free_energy_calcs(space, state_strs_screen, context)

            state_vecs = coupling.construct_state_vectors_single(indices, q_options)
            state_strs = utils.calc_state_strs(state_vecs)
            state_str0 = state_strs[0]  # Neutral state
            free_energy_diffs = {}
            for state_str1 in state_strs[1:]:
                free_energy_diffs[state_str1] = coupling.compare_free_energies(
                    indices,
                    q_options,
                    state_str0,
                    state_str1,
                    space.standard_free_energy_lib,
                )

            return coupling.construct_free_energy_coupling_weight_matrix(
                indices,
                state_strs,
                state_vecs,
                free_energy_diffs,
            )

        # Compare pKas if molgpka (no standard free energies directly)
        state_vecs = coupling.construct_state_vectors_single(indices, q_options)
        state_strs = utils.calc_state_strs(state_vecs)
        self.construct_mols(space, state_strs, state_vecs)
        self.run_acid_base_calcs(space, state_strs, state_vecs, context)

        base_pka_diffs = {}
        acid_pka_diffs = {}
        state_str0 = state_strs[0]  # Neutral state
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
        max_cut_edges: int = 1,
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
                    self.split_cluster_by_coupling_penalty(
                        component,
                        q_options0,
                        coupling_weights,
                        coupling_cutoff,max_cut_edges=max_cut_edges
                    )
                )
            return split_clusters

        cluster = components[0] if components else sorted(cluster)
        if count_state_combinations(q_options0[cluster]) <= self.cutoff_states:
            return [cluster]

        child_clusters = coupling.find_best_penalty_limited_split(
            graph,
            coupling_weights,
            lambda child_cluster: count_state_combinations(q_options0[child_cluster]),
            max_cut_edges,
            coupling_cutoff,
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
                        max_cut_edges=max_cut_edges
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
            max_cut_edges=max_cut_edges
        )

    def screen_clusters(
        self,
        indices0: list[int],
        q_options0: NDArray[np.int64],
        context: PredictorContext | None = None,
    ) -> list[list[int]]:
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

        phosphate_clusters = (
            self.phosphate_relative_clusters(indices0)
            if self.molgpka_prediction_mode(context)
            else []
        )

        if count_state_combinations(q_options0) <= self.cutoff_states and not phosphate_clusters:
            if len(indices0) == 0:
                return []
            return [list(range(len(indices0)))]

        coupling_cutoff = 0.1
        coupling_weights = self.coupling_assay_weights(indices0, q_options0, context=context)
        graph = coupling.coupling_weights_to_graph(coupling_weights, coupling_cutoff)
        clusters = [sorted(component) for component in nx.connected_components(graph)]

        split_clusters = []
        for cluster in sorted(clusters, key=lambda c: c[0]):
            split_clusters.extend(
                self.split_cluster_by_coupling_penalty(
                    cluster,
                    q_options0,
                    coupling_weights,
                    coupling_cutoff,
                    max_cut_edges=self.max_cut_edges
                )
            )
        return self.decouple_phosphate_clusters(split_clusters, phosphate_clusters)

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
    # Standard free energy calculation

    def run_standard_free_energy_calcs(
        self,
        space: ProtonationIndexSpace,
        state_strs: list[str],
        context: PredictorContext | None = None,
    ) -> None:
        """Compute and cache Uni-pKa standard free energies for microstates."""

        missing_state_strs = [
            state_str for state_str in state_strs if state_str not in space.standard_free_energy_lib
        ]
        if not missing_state_strs:
            return

        mols = [space.mols_lib[state_str] for state_str in missing_state_strs]
        standard_free_energies = self.predict_standard_free_energy_values(mols, context)
        for state_str, standard_free_energy in zip(missing_state_strs, standard_free_energies):
            space.standard_free_energy_lib[state_str] = standard_free_energy

    ###################################
    # Acid-base calculation

    def run_acid_base_calcs(
        self,
        space: ProtonationIndexSpace,
        state_strs: list[str],
        state_vecs: list[NDArray[np.int64]],
        context: PredictorContext | None = None,
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

            source_mol = space.mols_lib[state_str]

            state_vec_base = state_vec.copy()
            if self.opposite_charge_influence_mode(context):
                state_vec_base = state_vec
            else:
                # Disregard deprotonations of other sites to assess base probability.
                state_vec_base = np.maximum(state_vec, 1)

            state_str_base = pack_vec(state_vec_base)

            mol_base = space.mols_lib[state_str_base]

            base_tmp = self.pka_predictor(
                mol_base,
                context,
                source_mol=source_mol,
            ).pred_base()
            base = {}
            for map_idx, b in base_tmp.items():
                if map_idx not in space.indices:
                    continue
                rel_idx = space.indices.index(map_idx)
                if (
                    state_vec[rel_idx] == 1
                ):  # Only consider predicted protonation/de-protonation predictions from neutral state
                    base[map_idx] = b

            state_vec_acid = state_vec.copy()
            if self.opposite_charge_influence_mode(context):
                state_vec_acid = state_vec
            else:
                state_vec_acid = np.minimum(state_vec, 1)  # disregard protonations of other sites to assess acid probability
            state_str_acid = pack_vec(state_vec_acid)

            mol_acid = space.mols_lib[state_str_acid]

            acid_tmp = self.pka_predictor(
                mol_acid,
                context,
                source_mol=source_mol,
            ).pred_acid()

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
        state_freqs_sigmas_export: list[float | None] = []
        state_freqs_sigmas = (
            [None for _ in distribution.state_freqs]
            if distribution.state_freqs_sigmas is None
            else list(distribution.state_freqs_sigmas)
        )

        for state_str, state_freq, state_freq_sigma in zip(
            distribution.state_strs,
            distribution.state_freqs,
            state_freqs_sigmas,
        ):
            if state_freq >= self.cutoff_export * state_freq_max:  # Include all high prob states
                state_strs_export.append(state_str)
                state_freqs_export.append(state_freq)
                state_freqs_sigmas_export.append(state_freq_sigma)

        state_freqs_arr: NDArray[np.float64] = np.array(state_freqs_export)
        ps = np.argsort(state_freqs_arr)[::-1]  # Sort by highest probability

        state_freqs_export = [state_freqs_export[p] for p in ps]
        state_freqs_sigmas_export = [state_freqs_sigmas_export[p] for p in ps]
        state_strs_export = [state_strs_export[p] for p in ps]

        self.check_chiral_consistency(state_strs_export, distribution.indices)
        space = self.index_spaces.get(distribution.indices)

        molecule = combine_results(
            self.name,
            state_strs_export,
            state_freqs_export,
            space.mols_lib,
            distribution.state_qs,
            state_freqs_sigmas_export,
        )
        return molecule

    def check_chiral_consistency(
        self,
        state_strs: list[str],
        indices: list[int],
    ) -> None:
        """Retain the compatible stereochemistry of exported microstates.

        Each exported molecule is checked for 3D embeddability. If its atom
        stereochemistry is inconsistent, compatible centres are restored to a
        trial copy in priority order and only the incompatible constraints are
        omitted. The final isomeric SMILES is round-tripped and embedded before
        the validated molecule is written back to the cache.

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
            changed_map_indices = {
                map_idx
                for map_idx, state in zip(indices, unpack_vec(state_str))
                if state != 1
            }
            space.mols_lib[state_str] = relax_stereochemistry_for_embedding(
                mol,
                changed_map_indices,
            )
