"""Command-line entry: ``missions serve``, ``missions init``, ``missions doctor``."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.sandbox.executor import _bwrap_available
from src.settings import get_settings, reset_settings, resolve_home, save_settings
from src.settings.store import ensure_home, migrated_from_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="missions",
        description="Missions coding harness — serve the UI, inspect the install, init config.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Start the API and (if built) the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8088)
    serve.add_argument("--reload", action="store_true")

    init = sub.add_parser("init", help="Create ~/.missions (or TASK_CODER_HOME) and settings.json")
    init.add_argument("--force", action="store_true", help="Overwrite existing settings.json")

    sub.add_parser("doctor", help="Print install / config diagnostics")

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return _serve(args.host, args.port, args.reload)
    if args.cmd == "init":
        return _init(force=args.force)
    if args.cmd == "doctor":
        return _doctor()
    parser.print_help()
    return 1


def _serve(host: str, port: int, reload: bool) -> int:
    import uvicorn
    from src.api import create_app

    settings = get_settings()
    home = resolve_home()
    dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    print(f"[missions] home={home}")
    print(f"[missions] providers={len(settings.enabled_providers())} qdrant={settings.qdrant.mode}")
    if dist.is_dir():
        print(f"[missions] UI http://{host}:{port}/")
    else:
        print(
            f"[missions] API http://{host}:{port}/api/v1/health  "
            "(build the UI with: npm --prefix frontend run build)"
        )
    app = create_app()
    uvicorn.run(app, host=host, port=port, reload=reload)
    return 0


def _init(*, force: bool) -> int:
    home = ensure_home()
    from src.settings.store import settings_path
    path = settings_path(home)
    if path.exists() and not force:
        print(f"[missions] settings already exist at {path}")
        print("           pass --force to rewrite from defaults / .env")
        return 0
    settings = reset_settings()
    save_settings(settings)
    print(f"[missions] wrote {path}")
    print(f"[missions] home {home}")
    if migrated_from_env():
        print("[missions] imported values from the repository .env")
    return 0


def _row(name: str, value: str) -> None:
    print(f"  {name:<22} {value}")


def _doctor() -> int:
    home = resolve_home()
    settings = get_settings()
    dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    print("Missions doctor")
    print("===============")
    _row("python", sys.executable)
    _row("home", str(home))
    _row("settings", str(home / "settings.json"))
    _row("qdrant.mode", settings.qdrant.mode)
    _row("qdrant.path", settings.qdrant.path or str(home / "qdrant"))
    _row("embeddings", settings.embeddings.backend)
    _row("providers", str(len(settings.enabled_providers())) or "0")
    for provider in settings.llm.providers:
        flag = "on" if provider.enabled else "off"
        _row(f"  {provider.id}", f"{flag}  {provider.base_url}  model={provider.model or '(unset)'}")
    _row("bwrap", "yes" if _bwrap_available() else "no")
    _row("sandbox.executor", settings.runtime.sandbox.executor)
    _row("sandbox.jail", "bwrap" if _bwrap_available() else "policy-only")
    _row("require_bwrap", str(settings.runtime.sandbox.require_bwrap))
    _row("node", shutil.which("node") or "missing")
    _row("npm", shutil.which("npm") or "missing")
    _row("frontend dist", "yes" if (dist / "index.html").is_file() else "missing — run npm --prefix frontend run build")
    _row("git", shutil.which("git") or "missing")
    if Path("/.dockerenv").exists() or settings:
        in_docker = Path("/.dockerenv").exists()
        _row("container", "yes" if in_docker else "no")
        if in_docker:
            print()
            print("  Docker note: attach host folders only if they are bind-mounted.")
            print("  Use the container path (e.g. /host-projects/my-app) in the UI.")
    if not settings.enabled_providers():
        print()
        print("  No LLM providers configured. Open the UI wizard or Settings and")
        print("  add an OpenAI-compatible URL (llama.cpp / Ollama / LM Studio) or a cloud key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
