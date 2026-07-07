from pathlib import Path

from pkasso.api import chemistry
from pkasso.api.render import render_form
from pkasso.api.state import AppState, update_state_from_form
from pkasso.external.unipka.pka_predictor import FreeEnergyPredictionConfig


def test_precision_mode_is_rendered_and_restored_from_form():
    state = AppState(precision_mode=True)

    html = render_form(state)

    assert 'name="precision_mode"' in html
    assert "Precision Mode" in html

    update_state_from_form(
        state,
        {
            "ligand": "lig",
            "smiles": "C",
            "ph": "7.0",
            "nmols_export": "3",
            "tautomer_search": "on",
            "scan_enabled": "",
            "precision_mode": "",
        },
    )

    assert state.precision_mode is False

    update_state_from_form(
        state,
        {
            "ligand": "lig",
            "smiles": "C",
            "ph": "7.0",
            "nmols_export": "3",
            "tautomer_search": "on",
            "scan_enabled": "",
            "precision_mode": "on",
        },
    )

    assert state.precision_mode is True


def test_precision_mode_passes_unipka_model_and_config_to_protonate(monkeypatch):
    captured = {}
    model_dir = Path("/tmp/unipka-model")

    def fake_protonate(inp, **kwargs):
        captured["inp"] = inp
        captured.update(kwargs)
        return ("C",), ("mol",)

    monkeypatch.setattr(chemistry, "UNIPKA_MODEL_FOLDER", model_dir)
    monkeypatch.setattr(chemistry, "protonate", fake_protonate)

    state = AppState(smiles="C", precision_mode=True)
    chemistry.compute_prediction(state)

    assert captured["inp"] == "C"
    assert captured["model"] == ["molgpka", "unipka"]
    assert isinstance(captured["standard_free_energy_config"], FreeEnergyPredictionConfig)
    assert captured["standard_free_energy_config"].model_dir == model_dir
    assert captured["standard_free_energy_config"].batch_size == 16
    assert captured["standard_free_energy_config"].conf_size == 11


def test_precision_mode_passes_unipka_model_and_config_to_scan(monkeypatch):
    captured = {}
    model_dir = Path("/tmp/unipka-model")

    def fake_scan_ph(inp, **kwargs):
        captured["inp"] = inp
        captured.update(kwargs)
        return "scan"

    monkeypatch.setattr(chemistry, "UNIPKA_MODEL_FOLDER", model_dir)
    monkeypatch.setattr(chemistry, "scan_pH", fake_scan_ph)

    state = AppState(smiles="C", precision_mode=True)
    chemistry.compute_scan(state)

    assert state.scan == "scan"
    assert captured["inp"] == "C"
    assert captured["model"] == ["molgpka", "unipka"]
    assert isinstance(captured["standard_free_energy_config"], FreeEnergyPredictionConfig)
    assert captured["standard_free_energy_config"].model_dir == model_dir
