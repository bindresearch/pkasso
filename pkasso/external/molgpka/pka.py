"""molgpka pka calculations with custom rules"""
# mypy: disable-error-code=no-untyped-call

from pathlib import Path

import torch
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import Data

from .descriptor import MolVectorizer
from .ionization_group import get_ionization_aid
from .net import GCNNet

_DEFAULT_TORCH_NUM_THREADS = torch.get_num_threads()


def configure_torch_threads(nthreads: int) -> None:
    """Configure process-wide PyTorch CPU inference threads.

    PyTorch requires a positive intra-op thread count, so pKasso's public
    ``nthreads=0`` automatic setting restores the default captured at import.
    """

    if nthreads < 0:
        raise ValueError("nthreads must be at least 0.")
    torch.set_num_threads(_DEFAULT_TORCH_NUM_THREADS if nthreads == 0 else nthreads)


def load_model(model_file: Path) -> GCNNet:
    """Load molgpka ML torch model."""

    model = GCNNet().to("cpu")
    model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
    model.eval()
    return model


def model_pred_data(data: Data, model: GCNNet) -> float:
    """Predict pKa from precomputed molgpka graph data."""

    with torch.no_grad():
        data = data.to("cpu")
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka: float = pKa[0][0]
    return pka


def predict_acid(mol_h: Mol,
                 model_acid: GCNNet,
                 smarts_pattern: Path,
) -> dict[int, float]:
    """Predict acid pKas with molgpka model."""

    acid_idxs = get_ionization_aid(mol_h, "acid", smarts_pattern)
    acid_res = {}
    vectorizer = MolVectorizer(mol_h)
    for aid in acid_idxs:
        apka = model_pred_data(vectorizer.mol2vec(aid), model_acid)
        acid_res.update({aid: apka})
    return acid_res


def predict_base(mol_h: Mol,
                 model_base: GCNNet,
                 smarts_pattern: Path,
) -> dict[int, float]:
    """Predict base pKas with molgpka model."""

    base_idxs = get_ionization_aid(mol_h, "base", smarts_pattern)
    base_res = {}
    vectorizer = MolVectorizer(mol_h)
    for aid in base_idxs:
        bpka = model_pred_data(vectorizer.mol2vec(aid), model_base)
        base_res.update({aid: bpka})
    return base_res
