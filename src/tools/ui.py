"""Harness-managed local application serving and lightweight UI inspection."""

from __future__ import annotations

import html
import importlib.util
import re
import time
import uuid
import urllib.request
from typing import Any, Callable

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

_DYNAMIC_HTML_RE = re.compile(
    r"<script\b|type\s*=\s*['\"]module['\"]|id\s*=\s*['\"]root['\"]|id\s*=\s*['\"]app['\"]",
    re.I,
)
_FLOW_STEP_ACTIONS = {"fill", "click", "assert_text", "assert_visible", "snapshot", "wait"}


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


def _looks_dynamic_html(body: str) -> bool:
    return bool(_DYNAMIC_HTML_RE.search(body))


def _visible_text_from_html(body: str) -> str:
    visible = re.sub(r"<script\b[^>]*>.*?</script>", " ", body, flags=re.I | re.S)
    visible = re.sub(r"<style\b[^>]*>.*?</style>", " ", visible, flags=re.I | re.S)
    visible = html.unescape(re.sub(r"<[^>]+>", " ", visible))
    return " ".join(visible.split())


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


def _attach_page_listeners(page: Any) -> dict[str, list[str]]:
    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[str] = []
    http_failures: list[str] = []
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
        lambda response: http_failures.append(f"{response.status} {response.url}")
        if response.status >= 400
        and not response.url.rstrip("/").endswith("/favicon.ico")
        else None,
    )
    return {
        "console_errors": console_errors,
        "page_errors": page_errors,
        "failed_requests": failed_requests,
        "http_failures": http_failures,
    }


def _goto_page(page: Any, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=15_000)
    try:
        page.wait_for_load_state("networkidle", timeout=5_000)
    except Exception:
        pass


def _browser_visible_text(page: Any) -> str:
    try:
        text = page.locator("body").inner_text(timeout=5_000)
    except Exception:
        text = page.content()
    return " ".join(str(text).split())


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


def _playwright_missing(url: str) -> dict[str, Any]:
    return {
        "success": False,
        "url": url,
        "error": (
            "The browser backend is not installed. Install Playwright in "
            "the harness environment to enable this UI action."
        ),
        "capability_missing": "playwright",
    }


def _playwright_error(url: str, exc: Exception) -> dict[str, Any]:
    message = str(exc)
    lower_message = message.lower()
    result: dict[str, Any] = {
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


def _stateless_interaction_hint(action: str) -> str:
    return (
        f"This `{action}` call used a fresh browser session. State from a prior "
        "`fill` or `click` in a separate call is not preserved. For multi-step "
        "interactions, use `action: \"flow\"` with a `steps` array in one call."
    )


def _assert_text_handoff_hint() -> str:
    return (
        "Text was not found in the rendered page. If the UI looks correct, signal "
        "`complete` — the validator runs authoritative browser smoke checks next."
    )


def _run_flow_step(page: Any, step: dict[str, Any], *, url: str) -> dict[str, Any]:
    action = str(step.get("action", "")).lower().strip()
    if action not in _FLOW_STEP_ACTIONS:
        return {
            "success": False,
            "error": f"Unsupported flow step action '{action}'.",
            "step": step,
        }
    if action == "wait":
        page.wait_for_timeout(max(0, min(int(step.get("ms", 250)), 5_000)))
        return {"success": True, "action": "wait", "ms": step.get("ms", 250)}
    if action == "fill":
        selector = str(step.get("selector", ""))
        page.locator(selector).fill(str(step.get("value", "")), timeout=5_000)
        return {"success": True, "action": "fill", "selector": selector}
    if action == "click":
        selector = str(step.get("selector", ""))
        page.locator(selector).click(timeout=5_000)
        return {"success": True, "action": "click", "selector": selector}
    if action in {"assert_text", "assert_visible"}:
        expected = str(step.get("text", ""))
        visible = _browser_visible_text(page)
        passed = bool(expected) and expected in visible
        return {
            "success": passed,
            "action": action,
            "asserted_text": expected,
            "passed": passed,
            "visible_text": visible[:8000],
            "url": url,
            "hint": None if passed else _assert_text_handoff_hint(),
            "suggest_handoff": not passed,
        }
    visible = _browser_visible_text(page)
    return {
        "success": True,
        "action": "snapshot",
        "url": url,
        "visible_text": visible[:8000],
        "title": page.title(),
    }


def _with_browser(
    url: str,
    handler: Callable[[Any, dict[str, list[str]]], dict[str, Any]],
) -> dict[str, Any]:
    if importlib.util.find_spec("playwright") is None:
        return _playwright_missing(url)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            listeners = _attach_page_listeners(page)
            _goto_page(page, url)
            result = handler(page, listeners)
            browser.close()
            return result
    except Exception as exc:
        return _playwright_error(url, exc)


def _static_snapshot(url: str, *, action: str, text: str = "") -> dict[str, Any]:
    try:
        code, body = _fetch(url)
    except Exception as exc:
        return {"success": False, "url": url, "error": f"UI request failed: {exc}"}

    visible = _visible_text_from_html(body)
    dynamic = _looks_dynamic_html(body)
    result: dict[str, Any] = {
        "success": 200 <= code < 500,
        "url": url,
        "status_code": code,
        "visible_text": visible[:8000],
        "render_mode": "static_fetch",
        "title": (re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S) or [None, ""])[1].strip(),
    }
    if dynamic:
        result["hint"] = (
            "This page includes client-side JavaScript. Prefer browser actions "
            "(`accessibility`, `snapshot`, `assert_text`, or `flow`) so rendered "
            "content is evaluated."
        )
    if action == "assert_text":
        result["asserted_text"] = text
        result["passed"] = bool(text) and text in visible
        result["success"] = result["success"] and result["passed"]
        if dynamic and not result["passed"]:
            result["hint"] = (
                "assert_text via static fetch cannot see JavaScript-rendered content. "
                "Use `action: \"flow\"` for multi-step checks, or signal `complete` "
                "for validator smoke tests."
            )
            result["suggest_handoff"] = True
    return result


def inspect_ui(
    server_id: str,
    action: str = "navigate",
    path: str = "/",
    text: str = "",
    selector: str = "",
    value: str = "",
    steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Perform safe, text-oriented checks against a managed local server."""
    action = action.lower().strip()
    status = server_status(server_id)
    if not status.get("success") or not status.get("running"):
        return {"success": False, "error": "The requested server is not running."}

    url = f"http://127.0.0.1:{status['port']}{path if path.startswith('/') else '/' + path}"

    if action == "flow":
        flow_steps = [step for step in (steps or []) if isinstance(step, dict)]
        if not flow_steps:
            return {
                "success": False,
                "error": "action 'flow' requires a non-empty 'steps' array.",
                "url": url,
            }
        if len(flow_steps) > 12:
            flow_steps = flow_steps[:12]

        def _flow_handler(page: Any, listeners: dict[str, list[str]]) -> dict[str, Any]:
            step_results: list[dict[str, Any]] = []
            for step in flow_steps:
                step_result = _run_flow_step(page, step, url=url)
                step_results.append(step_result)
                if not step_result.get("success"):
                    diagnostics = _browser_diagnostics(**listeners)
                    return {
                        "success": False,
                        "url": url,
                        "render_mode": "browser",
                        "flow_results": step_results,
                        **diagnostics,
                        "hint": step_result.get("hint")
                        or "Fix the failing flow step or signal complete for validator smoke.",
                        "suggest_handoff": bool(step_result.get("suggest_handoff")),
                    }
            diagnostics = _browser_diagnostics(**listeners)
            return {
                "success": True,
                "url": url,
                "render_mode": "browser",
                "flow_results": step_results,
                **diagnostics,
            }

        return _with_browser(url, _flow_handler)

    if action in {"screenshot", "accessibility", "audit", "click", "fill"}:
        def _action_handler(page: Any, listeners: dict[str, list[str]]) -> dict[str, Any]:
            diagnostics = _browser_diagnostics(**listeners)
            if action == "screenshot":
                artifact_dir = _browser_artifact_dir()
                if artifact_dir is None:
                    return {"success": False, "url": url, "error": "No active sandbox context"}
                destination = artifact_dir / (
                    f"ui-{int(time.time())}-{uuid.uuid4().hex[:8]}.png"
                )
                page.screenshot(path=str(destination), full_page=True)
                return {
                    "success": True,
                    "url": url,
                    "render_mode": "browser",
                    "screenshot": str(destination),
                    **diagnostics,
                }
            if action == "accessibility":
                accessibility_issues = _basic_accessibility_issues(page)
                return {
                    "success": not accessibility_issues,
                    "url": url,
                    "render_mode": "browser",
                    "accessibility": page.locator("body").aria_snapshot(),
                    "accessibility_issues": accessibility_issues,
                    **diagnostics,
                }
            if action == "click":
                page.locator(selector).click(timeout=5_000)
                diagnostics = _browser_diagnostics(**listeners)
                return {
                    "success": True,
                    "url": url,
                    "render_mode": "browser",
                    "clicked": selector,
                    "hint": _stateless_interaction_hint("click"),
                    **diagnostics,
                }
            if action == "fill":
                page.locator(selector).fill(value, timeout=5_000)
                diagnostics = _browser_diagnostics(**listeners)
                return {
                    "success": True,
                    "url": url,
                    "render_mode": "browser",
                    "filled": selector,
                    "hint": _stateless_interaction_hint("fill"),
                    **diagnostics,
                }
            return {
                "success": not any(diagnostics.values()),
                "url": url,
                "render_mode": "browser",
                **diagnostics,
            }

        return _with_browser(url, _action_handler)

    if action in {"assert_text", "snapshot"}:
        if importlib.util.find_spec("playwright") is not None:
            def _render_handler(page: Any, listeners: dict[str, list[str]]) -> dict[str, Any]:
                visible = _browser_visible_text(page)
                diagnostics = _browser_diagnostics(**listeners)
                result: dict[str, Any] = {
                    "success": True,
                    "url": url,
                    "render_mode": "browser",
                    "visible_text": visible[:8000],
                    "title": page.title(),
                    **diagnostics,
                }
                if action == "assert_text":
                    passed = bool(text) and text in visible
                    result["asserted_text"] = text
                    result["passed"] = passed
                    result["success"] = passed
                    if not passed:
                        result["hint"] = _assert_text_handoff_hint()
                        result["suggest_handoff"] = True
                return result

            return _with_browser(url, _render_handler)
        return _static_snapshot(url, action=action, text=text)

    if action == "navigate":
        return _static_snapshot(url, action=action)

    return {
        "success": False,
        "error": (
            "Supported actions are navigate, snapshot, assert_text, flow, screenshot, "
            "accessibility, audit, click, and fill."
        ),
    }
