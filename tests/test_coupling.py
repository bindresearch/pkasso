import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_coupling_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "pkasso_coupling",
        root / "pkasso" / "coupling.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coupling = load_coupling_module()


def test_cluster_coupling_matrix_splits_single_bridge_between_dense_groups():
    matrix = np.zeros((6, 6), dtype=np.int64)
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 3),
    ]
    for i, j in edges:
        matrix[i, j] = 1

    assert coupling.cluster_coupling_matrix(matrix, bridge_cutoff=0) == [[0, 1, 2, 3, 4, 5]]
    assert coupling.cluster_coupling_matrix(matrix, bridge_cutoff=1) == [[0, 1, 2], [3, 4, 5]]


def test_bridge_cutoff_for_coupling_cutoff_schedule():
    assert coupling.bridge_cutoff_for_coupling_cutoff(0.1) == 0
    assert coupling.bridge_cutoff_for_coupling_cutoff(0.3) == 0
    assert coupling.bridge_cutoff_for_coupling_cutoff(0.4) == 1
    assert coupling.bridge_cutoff_for_coupling_cutoff(0.8) == 2
    assert coupling.bridge_cutoff_for_coupling_cutoff(1.2) == 3


def test_bridge_cutoff_validation():
    with pytest.raises(ValueError, match="bridge_cutoff must be non-negative"):
        coupling.cluster_coupling_matrix(np.zeros((2, 2), dtype=np.int64), bridge_cutoff=-1)

    with pytest.raises(ValueError, match="bridge_cutoff_step must be positive"):
        coupling.bridge_cutoff_for_coupling_cutoff(0.1, bridge_cutoff_step=0)
