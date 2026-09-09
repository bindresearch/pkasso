from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pkasso.api import chemistry
from pkasso.api.render import render_form, render_microstates, render_results
from pkasso.api.state import AppState, update_microstate_selection, update_state_from_form


def test_model_menu_is_rendered_and_restored_from_form():
    state = AppState(model="mixed")

    html = render_form(state)

    assert 'name="model"' in html
    assert "MolGpKa" in html
    assert "Uni-pKa" in html
    assert "Mixed" in html
    assert 'value="mixed" selected' in html
    assert "Precision Mode" not in html
    assert 'name="scan_enabled"' not in html
    assert 'name="ph"' not in html
    assert 'name="nmols_export"' not in html

    update_state_from_form(
        state,
        {
            "ligand": "lig",
            "smiles": "C",
            "tautomer_search": "on",
            "model": "unipka",
        },
    )
    assert state.model == "unipka"

    update_state_from_form(state, {"model": "unknown"})
    assert state.model == "molgpka"


@pytest.mark.parametrize(
    ("model_name", "expected_model", "uses_unipka_limits"),
    [
        ("molgpka", {"molgpka": {}}, False),
        ("unipka", {"unipka": "options"}, True),
        ("mixed", {"molgpka": {}, "unipka": "options"}, True),
    ],
)
def test_model_selection_is_passed_to_scan(
    monkeypatch,
    model_name,
    expected_model,
    uses_unipka_limits,
):
    captured = {}
    model_dir = Path("/tmp/unipka-model")

    def fake_scan_ph(inp, **kwargs):
        captured["inp"] = inp
        captured.update(kwargs)
        return "scan"

    monkeypatch.setattr(chemistry, "UNIPKA_MODEL_FOLDER", model_dir)
    monkeypatch.setattr(chemistry, "scan_pH", fake_scan_ph)

    state = AppState(smiles="C", model=model_name)
    chemistry.compute_scan(state)

    unipka_options = {"folds": (2,), "model_dir": model_dir}
    expected_model = {
        key: unipka_options if value == "options" else value
        for key, value in expected_model.items()
    }
    assert state.scan == "scan"
    assert captured["inp"] == "C"
    assert captured["model"] == expected_model
    assert captured["cutoff_export"] == 0.0
    assert "free_energy_cutoff_combined" not in captured
    assert captured["pHs"] == pytest.approx(np.arange(0, 14.05, 0.1))
    if uses_unipka_limits:
        assert captured["total_max_sites"] == 8
        assert captured["nthreads"] == 0
    else:
        assert "total_max_sites" not in captured
        assert "nthreads" not in captured


def test_lazy_microstate_selection_controls_output_count(monkeypatch):
    molecule = SimpleNamespace(
        smiles=("C", "[CH3+]", "[CH2-]"),
        mols=("mol-1", "mol-2", "mol-3"),
        freqs=(0.8, 0.0001, 0.000099),
    )
    calls = []
    scan = SimpleNamespace(molecule_at=lambda ph: calls.append(ph) or molecule)
    state = AppState(scan=scan)

    update_microstate_selection(state, "9.0", "3")
    chemistry.materialize_microstates(state)

    assert calls == [9.0]
    assert state.smiles_out == ["C", "[CH3+]"]
    assert state.mols_out == ["mol-1", "mol-2"]

    monkeypatch.setattr("pkasso.api.render.draw_molecule_grid", lambda *args, **kwargs: "<svg />")
    html = render_microstates(state, "/pkasso")
    assert "Predicted states at pH 9.0" in html
    assert "2 exported microstate(s)" in html
    assert "/pkasso/download/sdf?ph=9.0&amp;nmols_export=3" in html


def test_scan_results_render_plot_before_closed_lazy_details(monkeypatch):
    state = AppState(scan=SimpleNamespace(mols_relevant=["mol-1"]))
    monkeypatch.setattr("pkasso.api.render.scan_plot_svg", lambda *args, **kwargs: "<svg id='scan' />")
    monkeypatch.setattr("pkasso.api.render.draw_single_molecule", lambda *args, **kwargs: "<svg id='state' />")

    html = render_results(state)

    assert "<svg id='scan' />" in html
    assert "<svg id='state' />" in html
    assert 'hx-get="/scan/plot?highlight_idx=1"' in html
    assert "Hover a microstate image to highlight it" in html
    assert "xl:grid-cols-[minmax(0,1.35fr)_minmax(28rem,0.95fr)]" in html
    assert "<details" in html
    assert "Single-pH microstates" in html
    assert "Calculate states" in html
    assert 'hx-post="/microstates"' in html
    assert "loadMicrostates" not in html
    assert "input changed delay" not in html
    assert "Predicted states at pH" not in html
