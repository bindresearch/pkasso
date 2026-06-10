from __future__ import annotations

import argparse
import textwrap

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pKasso daisyUI/HTMX GUI.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--reload", action="store_true", help="Reload when local code changes.")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(textwrap.dedent(f"""
    pKasso GUI is running at {url}
    Press Ctrl+C to stop.
    """).strip())
    uvicorn.run("pkasso.api.web:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
