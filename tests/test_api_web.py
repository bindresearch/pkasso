import importlib.util

import pytest


WEB_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module) is not None
    for module in ("fastapi", "jinja2", "multipart", "itsdangerous")
)


@pytest.mark.skipif(not WEB_DEPENDENCIES_AVAILABLE, reason="requires pkasso[webserver]")
def test_precision_mode_preserves_optional_dependency_error():
    from pkasso.api.web import dependency_message

    exc = ModuleNotFoundError(
        "Uni-pKa support is not installed. Install it with "
        "`python -m pip install 'pkasso[unipka]'`."
    )

    assert dependency_message(exc) == str(exc)
