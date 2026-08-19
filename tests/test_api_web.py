import importlib.util

import pytest


WEB_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module) is not None
    for module in ("fastapi", "jinja2", "multipart", "itsdangerous")
)


@pytest.mark.skipif(not WEB_DEPENDENCIES_AVAILABLE, reason="requires pkasso[webserver]")
def test_model_selection_preserves_optional_dependency_error():
    from pkasso.api.web import dependency_message

    exc = ModuleNotFoundError(
        "Uni-pKa support is not installed. Install it with "
        "`python -m pip install 'pkasso[unipka]'`."
    )

    assert dependency_message(exc) == str(exc)


@pytest.mark.skipif(not WEB_DEPENDENCIES_AVAILABLE, reason="requires pkasso[webserver]")
def test_web_routes_expose_scan_submission_and_lazy_microstates():
    from pkasso.api.web import app

    routes = {
        (route.path, frozenset(getattr(route, "methods", ()) or ()))
        for route in app.routes
    }

    assert ("/predict", frozenset({"POST"})) in routes
    assert ("/microstates", frozenset({"POST"})) in routes
    assert ("/scan/plot", frozenset({"GET"})) in routes
    assert not any(path == "/scan" for path, _ in routes)
