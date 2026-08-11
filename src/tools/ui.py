"""Harness-managed local application serving and lightweight UI inspection."""

from __future__ import annotations

import html
import importlib.util
import re
import time
import uuid
import urllib.request
from typing import Any

from src.sandbox.context import get_sandbox_context
from src.sandbox.process_manager import (
    get_allowed_ports,
    list_servers,
    server_logs,
    server_status,
    start_server,
    stop_server,
)
from src.tools.paths import get_workspace_root, normalize_workspace_path


def _python(ctx: Any) -> str:
    return str(ctx.venv_python) if ctx is not None else "python"


def serve_app(
    action: str,
    kind: str = "auto",
    port: int = 9000,
    app_path: str = "app.py",
    entry_point: str = "",
    module: str = "app:app",
    command: list[str] | None = None,
    server_id: str = "",
    ready_url: str = "",
    timeout: int = 45,
) -> dict[str, Any]:
    """Start, inspect, log, list, or stop one harness-managed local server."""
    action = action.lower().strip()
    if action == "list":
        return list_servers()
    if action == "status":
        return server_status(server_id)
    if action == "logs":
        return server_logs(server_id)
    if action == "stop":
        return stop_server(server_id)
    if action != "start":
        return {
            "success": False,
            "error": "action must be start, list, status, logs, or stop",
            "allowed_ports": get_allowed_ports(),
        }

    ctx = get_sandbox_context()
    if ctx is None:
        return {"success": False, "error": "No active sandbox context"}
    root = get_workspace_root()
    selected = kind.lower().strip()
    if entry_point and app_path == "app.py":
        app_path = entry_point
    normalized_app = normalize_workspace_path(app_path)
    if selected == "auto":
        if (root / "package.json").exists():
            selected = "vite"
        elif (root / normalized_app).exists():
            selected = "streamlit"
        else:
            selected = "generic"

    if not ready_url:
        ready_url = f"http://127.0.0.1:{port}/"
    if selected in {"vite", "react", "next"}:
        argv = ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", str(port)]
    elif selected == "streamlit":
        argv = [
            _python(ctx), "-m", "streamlit", "run", normalized_app,
            "--server.address", "127.0.0.1",
            "--server.port", str(port),
            "--server.headless", "true",
        ]
    elif selected in {"fastapi", "uvicorn"}:
        argv = [
            _python(ctx), "-m", "uvicorn", module,
            "--host", "127.0.0.1", "--port", str(port),
        ]
    elif selected == "generic" and command:
        argv = list(command)
    else:
        return {
            "success": False,
            "error": "Provide a supported kind (vite, streamlit, fastapi, or generic command).",
        }

    return start_server(argv, port=port, ready_url=ready_url, timeout=timeout, ctx=ctx)


def _fetch(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=10) as response:
        body = response.read(200_000).decode("utf-8", errors="replace")
        return response.status, body


_BROWSER_ACTIONS = {"screenshot", "accessibility", "audit", "click", "fill"}


def _browser_artifact_dir() -> Any:
    ctx = get_sandbox_context()
    if ctx is None:
        return None
    artifact_dir = ctx.jail_root / "artifacts" / "ui"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _browser_diagnostics(
    console_errors: list[str],
    page_errors: list[str],
    failed_requests: list[str],
    http_failures: list[str],
) -> dict[str, Any]:
    return {
        "console_errors": console_errors[:50],
        "page_errors": page_errors[:50],
        "failed_requests": failed_requests[:50],
        "http_failures": http_failures[:50],
    }


def _basic_accessibility_issues(page: Any) -> list[str]:
    """Run dependency-free checks alongside Playwright's ARIA snapshot."""
    issues: list[str] = []
    if not str(page.title()).strip():
        issues.append("Document has no non-empty <title>.")
    unnamed_controls = page.locator(
        "button, a, input, select, textarea"
    ).evaluate_all(
        """elements => elements
          .filter(element => {
            const label = element.getAttribute('aria-label');
            const text = (element.innerText || element.value || '').trim();
            return !label && !text;
          })
          .slice(0, 20)
          .map(element => element.outerHTML.slice(0, 200))"""
    )
    issues.extend(
        f"Interactive element has no accessible name: {element}"
        for element in unnamed_controls
    )
    missing_alt = page.locator("img:not([alt])").count()
    if missing_alt:
        issues.append(f"{missing_alt} image(s) are missing an alt attribute.")
    return issues


def inspect_ui(
    server_id: str,
    action: str = "navigate",
    path: str = "/",
    text: str = "",
    selector: str = "",
    value: str = "",
) -> dict[str, Any]:
    """Perform safe, text-oriented checks against a managed local server."""
    status = server_status(server_id)
    if not status.get("success") or not status.get("running"):
        return {"success": False, "error": "The requested server is not running."}

    # The process manager only exposes loopback-bound servers. The path is
    # deliberately constrained to avoid turning this into an external fetcher.
    url = f"http://127.0.0.1:{status['port']}{path if path.startswith('/') else '/' + path}"
    if action in _BROWSER_ACTIONS:
        if importlib.util.find_spec("playwright") is None:
            return {
                "success": False,
                "url": url,
                "error": (
                    "The browser backend is not installed. Install Playwright in "
                    "the harness environment to enable this UI action."
                ),
                "capability_missing": "playwright",
            }
        try:
            from playwright.sync_api import sync_playwright

            console_errors: list[str] = []
            page_errors: list[str] = []
            failed_requests: list[str] = []
            http_failures: list[str] = []
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.on(
                    "console",
                    lambda message: console_errors.append(message.text)
                    if message.type == "error"
                    else None,
                )
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.on("requestfailed", lambda request: failed_requests.append(request.url))
                page.on(
                    "response",
                    lambda response: http_failures.append(
                        f"{response.status} {response.url}"
                    )
                    if response.status >= 400
                    and not response.url.rstrip("/").endswith("/favicon.ico")
                    else None,
                )
                page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                try:
                    page.wait_for_load_state("networkidle", timeout=5_000)
                except Exception:
                    # Apps with polling or websockets may never become idle.
                    pass
                diagnostics = _browser_diagnostics(
                    console_errors, page_errors, failed_requests, http_failures
                )
                if action == "screenshot":
                    artifact_dir = _browser_artifact_dir()
                    if artifact_dir is None:
                        return {"success": False, "url": url, "error": "No active sandbox context"}
                    destination = artifact_dir / (
                        f"ui-{int(time.time())}-{uuid.uuid4().hex[:8]}.png"
                    )
                    page.screenshot(path=str(destination), full_page=True)
                    result = {
                        "success": True,
                        "url": url,
                        "screenshot": str(destination),
                        **diagnostics,
                    }
                elif action == "accessibility":
                    accessibility_issues = _basic_accessibility_issues(page)
                    result = {
                        "success": not accessibility_issues,
                        "url": url,
                        "accessibility": page.locator("body").aria_snapshot(),
                        "accessibility_issues": accessibility_issues,
                        **diagnostics,
                    }
                elif action == "click":
                    page.locator(selector).click(timeout=5_000)
                    diagnostics = _browser_diagnostics(
                        console_errors, page_errors, failed_requests, http_failures
                    )
                    result = {"success": True, "url": url, "clicked": selector, **diagnostics}
                elif action == "fill":
                    page.locator(selector).fill(value, timeout=5_000)
                    diagnostics = _browser_diagnostics(
                        console_errors, page_errors, failed_requests, http_failures
                    )
                    result = {"success": True, "url": url, "filled": selector, **diagnostics}
                else:
                    result = {
                        "success": not any(diagnostics.values()),
                        "url": url,
                        **diagnostics,
                    }
                browser.close()
                return result
        except Exception as exc:
            message = str(exc)
            lower_message = message.lower()
            result = {
                "success": False,
                "url": url,
                "error": f"Browser inspection failed: {message}",
            }
            if (
                "executable doesn't exist" in lower_message
                or "playwright install" in lower_message
            ):
                result["capability_missing"] = "chromium"
                result["error"] = (
                    "Chromium is not installed for the harness Playwright runtime. "
                    "Run `python -m playwright install chromium` in the environment "
                    "that runs the API."
                )
            return result

    if action not in {"navigate", "assert_text", "snapshot"}:
        return {
            "success": False,
            "error": (
                "Supported actions are navigate, snapshot, assert_text, screenshot, "
                "accessibility, audit, click, and fill."
            ),
        }
    try:
        code, body = _fetch(url)
    except Exception as exc:
        return {"success": False, "url": url, "error": f"UI request failed: {exc}"}

    visible = re.sub(r"<script\b[^>]*>.*?</script>", " ", body, flags=re.I | re.S)
    visible = re.sub(r"<style\b[^>]*>.*?</style>", " ", visible, flags=re.I | re.S)
    visible = html.unescape(re.sub(r"<[^>]+>", " ", visible))
    visible = " ".join(visible.split())
    result: dict[str, Any] = {
        "success": 200 <= code < 500,
        "url": url,
        "status_code": code,
        "visible_text": visible[:8000],
        "title": (re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S) or [None, ""])[1].strip(),
    }
    if action == "assert_text":
        result["asserted_text"] = text
        result["passed"] = bool(text) and text in visible
        result["success"] = result["success"] and result["passed"]
    return result
