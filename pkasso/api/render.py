from __future__ import annotations

import html
from typing import Any

from .chemistry import draw_molecule_grid, draw_single_molecule, scan_figure_svg
from .state import AppState


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_empty(message: str) -> str:
    return f"""
    <div class="rounded-lg border border-dashed border-[color:var(--bind-border)] bg-white p-8 text-center text-sm text-[color:var(--bind-muted)]">
      {esc(message)}
    </div>
    """


def render_alert(message: str) -> str:
    return f"""
    <div role="alert" class="alert alert-error rounded-lg">
      <span>{esc(message)}</span>
    </div>
    """


def render_form(state: AppState) -> str:
    tautomer_checked = "checked" if state.tautomer_search else ""
    scan_checked = "checked" if state.scan_enabled else ""
    return f"""
    <form id="pkasso-form"
          class="space-y-5"
          hx-post="/predict"
          hx-target="#results"
          hx-swap="innerHTML"
          hx-indicator="#predict-indicator">
      <label class="form-control w-full">
        <div class="label px-0"><span class="label-text text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--bind-muted)]">Small molecule name</span></div>
        <input name="ligand" class="input input-bordered input-sm rounded-lg border-[color:var(--bind-border)] bg-white" value="{esc(state.ligand)}" />
      </label>

      <label class="form-control w-full">
        <div class="label px-0"><span class="label-text text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--bind-muted)]">SMILES code</span></div>
        <textarea name="smiles" class="textarea textarea-bordered min-h-28 rounded-lg border-[color:var(--bind-border)] bg-white font-mono text-sm">{esc(state.smiles)}</textarea>
      </label>

      <div class="grid grid-cols-1 gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,7rem)]">
        <div class="space-y-3">
          <label class="flex min-w-0 cursor-pointer items-center gap-3 rounded-lg border border-[color:var(--bind-border)] bg-[color:var(--bind-soft)] px-3 py-2">
            <input name="tautomer_search" type="checkbox" class="bind-toggle" {tautomer_checked} />
            <span class="min-w-0 text-sm font-medium">Tautomer search</span>
            <span class="bind-toggle-state ml-auto text-[11px] font-bold uppercase tracking-[0.12em] text-[color:var(--bind-muted)]"></span>
          </label>

          <label class="flex min-w-0 cursor-pointer items-center gap-3 rounded-lg border border-[color:var(--bind-border)] bg-[color:var(--bind-soft)] px-3 py-2">
            <input name="scan_enabled" type="checkbox" class="bind-toggle" {scan_checked} />
            <span class="min-w-0 text-sm font-medium">Full pH scan</span>
            <span class="bind-toggle-state ml-auto text-[11px] font-bold uppercase tracking-[0.12em] text-[color:var(--bind-muted)]"></span>
          </label>
        </div>

        <label class="form-control min-w-0">
          <div class="label px-0 py-1"><span class="label-text text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--bind-muted)]">Max states</span></div>
          <input name="nmols_export" type="number" min="1" max="20" value="{state.nmols_export}"
                 class="input input-bordered input-sm min-w-0 rounded-lg border-[color:var(--bind-border)] bg-white" />
        </label>
      </div>

      <div class="rounded-lg border border-[color:var(--bind-border)] bg-white p-4">
        <div class="mb-3 flex items-center justify-between gap-3">
          <span class="text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--bind-muted)]">pH</span>
          <output id="phValue" for="phRange"
                  class="min-w-14 rounded-lg bg-[color:var(--bind-soft)] px-2 py-1 text-center text-sm font-semibold text-[color:var(--bind-green)]">
            {state.ph:.1f}
          </output>
        </div>
        <input id="phRange" name="ph" type="range" min="0" max="14" step="0.1" value="{state.ph:.1f}"
               class="bind-range w-full"
               aria-describedby="phValue"
               data-ph-range
               style="--ph-position: {state.ph / 14 * 100:.3f}" />
        <div class="mt-2 grid grid-cols-3 text-[11px] font-medium text-[color:var(--bind-muted)]">
          <span>0</span><span class="text-center">7</span><span class="text-right">14</span>
        </div>
      </div>

      <button type="submit" class="btn btn-primary btn-sm w-full rounded-lg">
        <span style='color:white;'>Calculate states</span>
        <span id="predict-indicator" class="loading loading-spinner loading-xs htmx-indicator"></span>
      </button>
    </form>
    """


def render_results(state: AppState) -> str:
    if state.error:
        return render_alert(state.error)

    if not state.smiles_out:
        return render_empty("Enter a SMILES string and calculate states to begin.")

    smiles_lines = "\n".join(state.smiles_out)
    try:
        molecule_grid = draw_molecule_grid(state.mols_out, show_probability=True)
    except Exception as exc:  # pragma: no cover - visual helper failure path
        molecule_grid = render_alert(f"Could not render molecule images: {exc}")

    scan_html = (
        render_scan(state)
        if state.scan is not None
        else render_empty("Enable pH scan and calculate states to inspect microstate distributions.")
    )
    return f"""
    <section class="space-y-4">
      <div class="flex flex-wrap items-start justify-between gap-3 border-b border-[color:var(--bind-border)] pb-4">
        <div>
          <p class="section-kicker">Single pH prediction</p>
          <h2 class="mt-1 text-2xl font-semibold tracking-tight">Predicted states at pH {state.ph:.1f}</h2>
          <p class="mt-1 text-sm text-[color:var(--bind-muted)]">{len(state.mols_out)} exported microstate(s)</p>
        </div>
        <div class="flex flex-wrap gap-2">
          <a class="btn btn-outline btn-sm rounded-lg" href="/download/sdf">Download SDF</a>
          <button type="button" class="btn btn-secondary btn-sm rounded-lg" data-feedback-open>Feedback</button>
        </div>
      </div>

      <pre class="overflow-x-auto rounded-lg border border-[color:var(--bind-border)] bg-[color:var(--bind-soft)] p-4 text-sm text-[#32716D]"><code>{esc(smiles_lines)}</code></pre>

      <div class="overflow-x-auto rounded-lg border border-[color:var(--bind-border)] bg-white p-3 [&_svg]:max-w-full [&_svg]:h-auto">
        {molecule_grid}
      </div>

      <div id="scan-panel">
        {scan_html}
      </div>
    </section>
    """


def render_scan(state: AppState) -> str:
    if state.scan is None:
        return render_empty("Enable pH scan and calculate states to inspect microstate distributions.")

    mols_relevant = list(getattr(state.scan, "mols_relevant", []))
    plot = scan_plot_svg(state, 0)
    microstates = []
    for idx, mol in enumerate(mols_relevant):
        try:
            mol_svg = draw_single_molecule(mol)
        except Exception as exc:  # pragma: no cover - visual helper failure path
            mol_svg = render_alert(f"Could not render microstate {idx + 1}: {exc}")
        microstates.append(
            f"""
            <button type="button"
                    class="microstate rounded-lg border border-[color:var(--bind-border)] bg-white p-3 text-left transition hover:border-accent hover:bg-accent/5 focus:outline-none focus:ring-2 focus:ring-accent"
                    data-microstate-enlarge
                    data-microstate-title="Microstate {idx + 1}"
                    hx-get="/scan/plot?highlight_idx={idx + 1}"
                    hx-target="#scan-plot"
                    hx-swap="innerHTML"
                    hx-trigger="mouseenter, focus, click">
              <span class="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-[color:var(--bind-muted)]">Microstate {idx + 1}</span>
              <span class="microstate-image block [&_svg]:h-auto [&_svg]:max-w-full">{mol_svg}</span>
            </button>
            """
        )

    if not microstates:
        microstate_html = render_empty("No relevant microstates were returned for this scan.")
    else:
        grid_cols = "2xl:grid-cols-3" if len(microstates) > 4 else ""
        microstate_html = f"""
        <div class="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-2 {grid_cols}">
          {"".join(microstates)}
        </div>
        """

    return f"""
    <section class="rounded-lg border border-[color:var(--bind-border)] bg-white p-4">
      <div class="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p class="section-kicker">Full pH scan</p>
          <h2 class="mt-1 text-2xl font-semibold tracking-tight">Microstate distributions</h2>
          <p class="mt-1 text-sm text-[color:var(--bind-muted)]">Hover a microstate image to highlight it in the distribution plot; click to enlarge it.</p>
        </div>
        <button type="button"
                class="btn btn-ghost btn-sm rounded-lg"
                hx-get="/scan/plot?highlight_idx=0"
                hx-target="#scan-plot"
                hx-swap="innerHTML">
          Clear highlight
        </button>
      </div>
      <div class="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1.35fr)_minmax(28rem,0.95fr)]">
        <div id="scan-plot" class="rounded-lg border border-[color:var(--bind-border)] bg-white p-4 [&_svg]:h-auto [&_svg]:w-full [&_svg]:max-w-full">
          {plot}
        </div>
        {microstate_html}
      </div>
    </section>
    """


def scan_plot_svg(state: AppState, highlight_idx: int) -> str:
    plot = scan_figure_svg(state, highlight_idx)
    if plot is None:
        return render_empty("Run a pH scan to show the distribution plot.")
    return plot