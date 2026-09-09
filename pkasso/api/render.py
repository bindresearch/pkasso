from __future__ import annotations

import html
from typing import Any

from .chemistry import draw_molecule_grid, draw_single_molecule, scan_figure_svg
from .state import AppState


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def prefixed_path(root_path: str, path: str) -> str:
    root_path = root_path.rstrip("/")
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{root_path}{path}"


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


def render_warnings(messages: list[str]) -> str:
    if not messages:
        return ""

    items = "".join(f"<li>{esc(message)}</li>" for message in messages)
    return f"""
    <div role="alert" class="alert alert-warning items-start rounded-lg">
      <div>
        <p class="font-semibold">pKasso reported a warning</p>
        <ul class="mt-1 list-disc space-y-1 pl-5 text-sm">{items}</ul>
      </div>
    </div>
    """


def render_form(state: AppState, root_path: str = "") -> str:
    tautomer_checked = "checked" if state.tautomer_search else ""
    predict_url = prefixed_path(root_path, "/predict")
    model_options = "".join(
        f'<option value="{value}" {"selected" if state.model == value else ""}>{label}</option>'
        for value, label in (
            ("molgpka", "MolGpKa"),
            ("unipka", "Uni-pKa"),
            ("mixed", "Mixed"),
        )
    )
    return f"""
    <form id="pkasso-form"
          class="space-y-5"
          hx-post="{predict_url}"
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

      <label class="form-control w-full">
        <div class="label px-0 py-1"><span class="label-text text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--bind-muted)]">Model</span></div>
        <select name="model" class="select select-bordered select-sm w-full rounded-lg border-[color:var(--bind-border)] bg-white">
          {model_options}
        </select>
      </label>

      <label class="flex min-w-0 cursor-pointer items-center gap-3 rounded-lg border border-[color:var(--bind-border)] bg-[color:var(--bind-soft)] px-3 py-2">
        <input name="tautomer_search" type="checkbox" class="bind-toggle" {tautomer_checked} />
        <span class="min-w-0 text-sm font-medium">Tautomer search</span>
        <span class="bind-toggle-state ml-auto text-[11px] font-bold uppercase tracking-[0.12em] text-[color:var(--bind-muted)]"></span>
      </label>

      <button type="submit" class="btn btn-primary btn-sm w-full rounded-lg">
        <span style='color:white;'>Run pH scan</span>
        <span id="predict-indicator" class="loading loading-spinner loading-xs htmx-indicator"></span>
      </button>
    </form>
    """


def render_results(state: AppState, root_path: str = "") -> str:
    if state.error:
        return render_alert(state.error)

    if state.scan is None:
        return render_empty("Enter a SMILES string and run a pH scan to begin.")

    warnings_html = render_warnings(state.warnings)
    return f"""
    <section class="space-y-4">
      {warnings_html}
      {render_scan(state, root_path)}
    </section>
    """


def render_scan(state: AppState, root_path: str = "") -> str:
    if state.scan is None:
        return render_empty("Run a pH scan to inspect microstate distributions.")

    plot = scan_plot_svg(state, 0)
    max_state_options = "".join(
        f'<option value="{value}" {"selected" if state.nmols_export == value else ""}>{value}</option>'
        for value in range(1, 21)
    )
    microstates_url = prefixed_path(root_path, "/microstates")
    model_label = {
        "molgpka": "MolGpKa",
        "unipka": "Uni-pKa",
        "mixed": "Mixed",
    }[state.model]
    microstates = []
    for idx, mol in enumerate(getattr(state.scan, "mols_relevant", [])):
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
                    hx-get="{prefixed_path(root_path, f'/scan/plot?highlight_idx={idx + 1}')}"
                    hx-target="#scan-plot"
                    hx-swap="innerHTML"
                    hx-trigger="mouseenter, focus, click">
              <span class="mb-2 block text-xs font-semibold uppercase tracking-[0.12em] text-[color:var(--bind-muted)]">Microstate {idx + 1}</span>
              <span class="microstate-image block [&_svg]:h-auto [&_svg]:max-w-full">{mol_svg}</span>
            </button>
            """
        )

    if microstates:
        grid_cols = "2xl:grid-cols-3" if len(microstates) > 4 else ""
        microstate_html = f"""
        <div class="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-2 {grid_cols}">
          {"".join(microstates)}
        </div>
        """
    else:
        microstate_html = render_empty("No relevant microstates were returned for this scan.")

    return f"""
    <section class="rounded-lg border border-[color:var(--bind-border)] bg-white p-4">
      <div class="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p class="section-kicker">Full pH scan</p>
          <h2 class="mt-1 text-2xl font-semibold tracking-tight">Microstate distributions</h2>
          <p class="mt-1 text-sm text-[color:var(--bind-muted)]">Hover a microstate image to highlight it in the distribution plot; click to enlarge it.</p>
          <p class="mt-1 text-xs text-[color:var(--bind-muted)]">Model: {esc(model_label)}</p>
        </div>
        <div class="flex flex-wrap gap-2">
          <button type="button"
                  class="btn btn-ghost btn-sm rounded-lg"
                  hx-get="{prefixed_path(root_path, '/scan/plot?highlight_idx=0')}"
                  hx-target="#scan-plot"
                  hx-swap="innerHTML">
            Clear highlight
          </button>
          <button type="button" class="btn btn-secondary btn-sm rounded-lg" data-feedback-open>Feedback</button>
        </div>
      </div>
      <div class="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1.35fr)_minmax(28rem,0.95fr)]">
        <div id="scan-plot" class="rounded-lg border border-[color:var(--bind-border)] bg-white p-4 [&_svg]:h-auto [&_svg]:w-full [&_svg]:max-w-full">
          {plot}
        </div>
        {microstate_html}
      </div>

      <details class="mt-4 rounded-lg border border-[color:var(--bind-border)] bg-[color:var(--bind-soft)]" data-microstate-expander>
        <summary class="cursor-pointer px-4 py-3 text-sm font-semibold">Single-pH microstates</summary>
        <div class="space-y-4 border-t border-[color:var(--bind-border)] p-4">
          <form class="grid grid-cols-1 gap-4 md:grid-cols-[minmax(0,1fr)_9rem]"
                hx-post="{microstates_url}"
                hx-target="#microstate-output"
                hx-swap="innerHTML"
                hx-indicator="#microstate-indicator">
            <div class="rounded-lg border border-[color:var(--bind-border)] bg-white p-4">
              <div class="mb-3 flex items-center justify-between gap-3">
                <span class="text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--bind-muted)]">pH</span>
                <output id="phValue" for="phRange" class="min-w-14 rounded-lg bg-[color:var(--bind-soft)] px-2 py-1 text-center text-sm font-semibold text-[color:var(--bind-green)]">{state.ph:.1f}</output>
              </div>
              <input id="phRange" name="ph" type="range" min="0" max="14" step="0.1" value="{state.ph:.1f}"
                     class="bind-range w-full" aria-describedby="phValue" data-ph-range
                     style="--ph-position: {state.ph / 14 * 100:.3f}" />
              <div class="mt-2 grid grid-cols-3 text-[11px] font-medium text-[color:var(--bind-muted)]">
                <span>0</span><span class="text-center">7</span><span class="text-right">14</span>
              </div>
            </div>
            <label class="form-control min-w-0">
              <div class="label px-0 py-1"><span class="label-text text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--bind-muted)]">Max states</span></div>
              <select name="nmols_export" class="select select-bordered select-sm w-full rounded-lg border-[color:var(--bind-border)] bg-white">
                {max_state_options}
              </select>
            </label>
            <button type="submit" class="btn btn-primary btn-sm rounded-lg md:col-span-2">
              <span style="color:white;">Calculate states</span>
              <span id="microstate-indicator" class="loading loading-spinner loading-xs htmx-indicator"></span>
            </button>
          </form>
          <div id="microstate-output">
            {render_empty("Choose a pH to generate microstates.")}
          </div>
        </div>
      </details>
    </section>
    """


def render_microstates(state: AppState, root_path: str = "") -> str:
    if not state.mols_out:
        return render_empty("No microstates were returned at this pH.")

    smiles_lines = "\n".join(state.smiles_out)
    try:
        molecule_grid = draw_molecule_grid(state.mols_out, show_probability=True)
    except Exception as exc:  # pragma: no cover - visual helper failure path
        molecule_grid = render_alert(f"Could not render molecule images: {exc}")

    download_url = prefixed_path(
        root_path,
        f"/download/sdf?ph={state.ph:.1f}&amp;nmols_export={state.nmols_export}",
    )
    return f"""
    <section class="space-y-3 rounded-lg border border-[color:var(--bind-border)] bg-white p-4">
      <div class="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 class="text-lg font-semibold">Predicted states at pH {state.ph:.1f}</h3>
          <p class="text-sm text-[color:var(--bind-muted)]">{len(state.mols_out)} exported microstate(s)</p>
        </div>
        <a class="btn btn-outline btn-sm rounded-lg" href="{download_url}">Download SDF</a>
      </div>
      <pre class="overflow-x-auto rounded-lg border border-[color:var(--bind-border)] bg-[color:var(--bind-soft)] p-4 text-sm text-[#32716D]"><code>{esc(smiles_lines)}</code></pre>
      <div class="overflow-x-auto rounded-lg border border-[color:var(--bind-border)] bg-white p-3 [&_svg]:h-auto [&_svg]:max-w-full">
        {molecule_grid}
      </div>
    </section>
    """


def scan_plot_svg(state: AppState, highlight_idx: int) -> str:
    plot = scan_figure_svg(state, highlight_idx)
    if plot is None:
        return render_empty("Run a pH scan to show the distribution plot.")
    return plot
