"""
Run the Missions Control API via uvicorn.

Usage::

    python -m src.api               # default :8088
    python -m src.api --host 0.0.0.0 --port 8088
"""

from __future__ import annotations

import argparse

import uvicorn

from src.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Missions Control API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8088, help="Bind port (default: 8088)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    parser.add_argument("log_level", choices=["debug", "info", "warning", "error", "critical"], default="error", help="Log level (default: info)")
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
