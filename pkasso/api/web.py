from __future__ import annotations

import os
import uuid
from typing import Annotated

from fastapi import FastAPI, Form, Query, Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .chemistry import compute_scan, materialize_microstates, sdf_for_state
from .config import DEFAULT_LIGAND, DEFAULT_SMILES, STATIC_DIR, TEMPLATE_DIR
from .feedback import clean_feedback, save_feedback
from .render import render_alert, render_form, render_microstates, render_results, scan_plot_svg
from .state import AppState, SESSIONS, update_microstate_selection, update_state_from_form


app = FastAPI(title="pKasso", docs_url=None, redoc_url=None)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("PKASSO_SESSION_SECRET", "pkasso-local-dev"),
    same_site="lax",
    https_only=False,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATE_DIR)


def request_root_path(request: Request) -> str:
    return str(request.scope.get("root_path") or "").rstrip("/")


def dependency_message(exc: Exception) -> str:
    detail = str(exc)
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        return (
            f"Missing Python dependency: {exc.name}. Install the project dependencies "
            "with `pip install .` in the environment that runs this server."
        )
    return detail or exc.__class__.__name__


def session_state(request: Request) -> AppState:
    sid = request.session.get("ap_session")
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex
        request.session["ap_session"] = sid
        SESSIONS[sid] = AppState()
    return SESSIONS[sid]


def form_payload(
    ligand: str,
    smiles: str,
    tautomer_search: str | None,
    model: str,
) -> dict[str, str]:
    return {
        "ligand": ligand,
        "smiles": smiles,
        "tautomer_search": "on" if tautomer_search else "",
        "model": model,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Response:
    state = session_state(request)
    root_path = request_root_path(request)
    return templates.TemplateResponse(
        request=request,
        name="page.html",
        context={
            "form": render_form(state, root_path),
            "results": render_results(state, root_path),
            "root_path": root_path,
        },
    )


@app.get("/download/sdf")
async def download_sdf(
    request: Request,
    ph: Annotated[str, Query()] = "7.0",
    nmols_export: Annotated[str, Query()] = "3",
) -> Response:
    state = session_state(request)
    if state.scan is None:
        return PlainTextResponse(
            "No pH scan has been calculated yet.",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    try:
        update_microstate_selection(state, ph, nmols_export)
        materialize_microstates(state)
        data = sdf_for_state(state)
    except Exception as exc:
        return PlainTextResponse(
            f"Could not generate SDF: {exc}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    filename = f"{state.ligand or DEFAULT_LIGAND}.sdf"
    return Response(
        data,
        media_type="chemical/x-mdl-sdfile",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/scan/plot", response_class=HTMLResponse)
async def scan_plot(
    request: Request,
    highlight_idx: Annotated[str, Query()] = "0",
) -> HTMLResponse:
    state = session_state(request)
    try:
        selected_idx = int(highlight_idx)
    except ValueError:
        selected_idx = 0
    return HTMLResponse(scan_plot_svg(state, selected_idx))


@app.post("/feedback", response_class=HTMLResponse)
async def feedback(
    smiles: Annotated[str, Form()] = "",
    comment: Annotated[str, Form()] = "",
) -> HTMLResponse:
    try:
        smiles, comment = clean_feedback(smiles, comment)
        save_feedback(smiles, comment)
    except ValueError as exc:
        return HTMLResponse(render_alert(str(exc)), status_code=status.HTTP_400_BAD_REQUEST)
    except Exception:
        return HTMLResponse(
            render_alert("Could not save feedback."),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return HTMLResponse(
        """
        <div role="status" class="alert alert-success rounded-lg" style="background-color:var(--bind-yellow);border:none"} data-feedback-saved>
          <span>Feedback saved. Thank you.</span>
        </div>
        """
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    ligand: Annotated[str, Form()] = DEFAULT_LIGAND,
    smiles: Annotated[str, Form()] = DEFAULT_SMILES,
    tautomer_search: Annotated[str | None, Form()] = None,
    model: Annotated[str, Form()] = "molgpka",
) -> HTMLResponse:
    state = session_state(request)
    root_path = request_root_path(request)
    update_state_from_form(
        state,
        form_payload(
            ligand,
            smiles,
            tautomer_search,
            model,
        ),
    )

    if not state.smiles:
        state.error = "Please enter a SMILES string."
        return HTMLResponse(render_results(state), status_code=status.HTTP_400_BAD_REQUEST)

    try:
        compute_scan(state)
    except Exception as exc:
        state.error = dependency_message(exc)

    return HTMLResponse(render_results(state, root_path))


@app.post("/microstates", response_class=HTMLResponse)
async def microstates(
    request: Request,
    ph: Annotated[str, Form()] = "7.0",
    nmols_export: Annotated[str, Form()] = "3",
) -> HTMLResponse:
    state = session_state(request)
    root_path = request_root_path(request)
    update_microstate_selection(state, ph, nmols_export)
    if state.scan is None:
        return HTMLResponse(
            render_alert("Run a pH scan before selecting microstates."),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    try:
        materialize_microstates(state)
    except Exception as exc:
        return HTMLResponse(render_alert(dependency_message(exc)))

    return HTMLResponse(render_microstates(state, root_path))


@app.exception_handler(404)
async def not_found(_: Request, __: Exception) -> PlainTextResponse:
    return PlainTextResponse("Not found", status_code=status.HTTP_404_NOT_FOUND)
