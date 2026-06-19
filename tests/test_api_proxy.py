from pathlib import Path

from pkasso.api.app import normalize_forwarded_allow_ips, normalize_root_path


def test_normalize_root_path():
    assert normalize_root_path("") == ""
    assert normalize_root_path("/") == ""
    assert normalize_root_path("pkasso") == "/pkasso"
    assert normalize_root_path("/pkasso/") == "/pkasso"


def test_normalize_forwarded_allow_ips():
    assert normalize_forwarded_allow_ips(None) is None
    assert normalize_forwarded_allow_ips("") is None
    assert normalize_forwarded_allow_ips("  ") is None
    assert normalize_forwarded_allow_ips("*") == "*"
    assert normalize_forwarded_allow_ips("10.0.0.1,10.0.0.2") == "10.0.0.1,10.0.0.2"


def test_template_static_assets_are_path_only():
    template = Path("pkasso/api/templates/page.html").read_text()

    assert "url_for('static'" not in template
    assert "{{ root_path }}/static/app.css" in template
    assert "{{ root_path }}/static/app.js" in template
