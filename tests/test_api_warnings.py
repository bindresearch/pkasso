import logging

from pkasso.api import chemistry
from pkasso.api.render import render_results
from pkasso.api.state import AppState


def test_compute_scan_collects_core_warnings(monkeypatch):
    def fake_scan(inp, **kwargs):
        logger = logging.getLogger("pkasso.main")
        logger.warning("Molecule has >8 protonation sites. Returning processed input molecule.")
        logger.warning("Molecule has >8 protonation sites. Returning processed input molecule.")
        return object()

    monkeypatch.setattr(chemistry, "scan_pH", fake_scan)
    state = AppState(smiles="C")

    chemistry.compute_scan(state)

    assert state.warnings == [
        "Molecule has >8 protonation sites. Returning processed input molecule."
    ]


def test_compute_scan_clears_warnings_when_next_run_has_none(monkeypatch):
    monkeypatch.setattr(chemistry, "scan_pH", lambda inp, **kwargs: object())
    state = AppState(smiles="C", warnings=["old warning"])

    chemistry.compute_scan(state)

    assert state.warnings == []


def test_render_results_displays_and_escapes_warnings(monkeypatch):
    monkeypatch.setattr("pkasso.api.render.scan_plot_svg", lambda *args, **kwargs: "")
    state = AppState(
        scan=object(),
        warnings=["Several fragments contain <carbon> & other atoms."],
    )

    html = render_results(state)

    assert "pKasso reported a warning" in html
    assert "Several fragments contain &lt;carbon&gt; &amp; other atoms." in html
