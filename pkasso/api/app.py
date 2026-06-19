from __future__ import annotations

import argparse
import importlib.util
import os

WEB_DEPENDENCIES = {
    "fastapi": "fastapi",
    "jinja2": "jinja2",
    "multipart": "python-multipart",
    "itsdangerous": "itsdangerous",
    "uvicorn": "uvicorn",
}


def normalize_root_path(value: str) -> str:
    value = value.strip()
    if not value or value == "/":
        return ""
    return "/" + value.strip("/")


def normalize_forwarded_allow_ips(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pKasso daisyUI/HTMX GUI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--reload", action="store_true", help="Reload when local code changes.")
    parser.add_argument(
        "--root-path",
        default=os.environ.get("PKASSO_ROOT_PATH", ""),
        help="Public URL path prefix when served behind a proxy, e.g. /pkasso. Defaults to PKASSO_ROOT_PATH.",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default=os.environ.get("PKASSO_FORWARDED_ALLOW_IPS", os.environ.get("FORWARDED_ALLOW_IPS")),
        help=(
            "Comma-separated proxy IPs whose X-Forwarded-* headers should be trusted. "
            "Use '*' only when pKasso is reachable exclusively through a trusted proxy. "
            "Defaults to PKASSO_FORWARDED_ALLOW_IPS or FORWARDED_ALLOW_IPS."
        ),
    )
    args = parser.parse_args()
    args.root_path = normalize_root_path(args.root_path)
    args.forwarded_allow_ips = normalize_forwarded_allow_ips(args.forwarded_allow_ips)

    missing = [
        package_name for module_name, package_name in WEB_DEPENDENCIES.items() if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        parser.exit(
            1,
            "The pKasso web server dependencies are not installed.\n"
            f"Missing: {', '.join(sorted(set(missing)))}.\n"
            "Install them with: pip install 'pkasso[webserver]'\n",
        )

    import uvicorn

    uvicorn.run(
        "pkasso.api.web:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        proxy_headers=True,
        forwarded_allow_ips=args.forwarded_allow_ips,
        root_path=args.root_path,
    )


if __name__ == "__main__":
    main()
