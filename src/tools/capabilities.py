"""Capability-oriented project inspection and verification tools.

These tools intentionally expose a small, language-neutral surface. The
harness selects concrete commands from project manifests instead of asking a
small model to remember one tool per language.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from src.sandbox.context import get_sandbox_context
from src.sandbox.executor import get_executor
from src.sandbox.policy import NetworkMode
from src.tools.paths import get_workspace_root


_MANIFEST_ECOSYSTEMS = {
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "package.json": "node",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "pom.xml": "jvm",
    "build.gradle": "jvm",
    "build.gradle.kts": "jvm",
    "Makefile": "generic",
    "CMakeLists.txt": "generic",
}


def _workspace() -> Path:
    return get_workspace_root().resolve()


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root))


def _detect_ecosystems(root: Path) -> tuple[list[str], list[str]]:
    manifests: list[str] = []
    ecosystems: set[str] = set()
    for name, ecosystem in _MANIFEST_ECOSYSTEMS.items():
        path = root / name
        if path.exists():
            manifests.append(name)
            ecosystems.add(ecosystem)
    if (root / ".streamlit").is_dir():
        manifests.append(".streamlit/")
        ecosystems.add("streamlit")
    for path in root.glob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "import streamlit" in text or "from streamlit" in text:
            ecosystems.add("streamlit")
            break
    return sorted(ecosystems), sorted(manifests)


def project_info(max_entries: int = 80) -> dict[str, Any]:
    """Return a bounded, deterministic summary of the current workspace."""
    root = _workspace()
    ecosystems, manifests = _detect_ecosystems(root)
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if len(entries) >= max(1, min(max_entries, 200)):
            break
        if any(part in {".git", ".venv", "__pycache__", "node_modules", "target"} for part in path.parts):
            continue
        try:
            entries.append(_relative(path, root) + ("/" if path.is_dir() else ""))
        except ValueError:
            continue

    toolchains = {
        name: bool(shutil.which(name))
        for name in ("python", "node", "npm", "pnpm", "go", "cargo", "rustc", "java", "mvn", "gradle")
    }
    return {
        "success": True,
        "workspace": str(root),
        "ecosystems": ecosystems,
        "manifests": manifests,
        "entries": entries,
        "toolchains": toolchains,
        "active_sandbox": get_sandbox_context() is not None,
    }


def _commands_for(ecosystem: str, checks: list[str], target: str) -> list[tuple[str, list[str]]]:
    """Compile requested checks to argv without a shell."""
    target = target or "."
    selected = set(checks or ["test"])
    if ecosystem == "python":
        commands = {
            "test": ["python", "-m", "pytest", target, "-q"],
            "lint": ["python", "-m", "flake8", target],
            "typecheck": ["python", "-m", "mypy", target],
            "build": ["python", "-m", "compileall", "-q", target],
        }
    elif ecosystem == "node":
        commands = {
            "test": ["npm", "test", "--", target],
            "lint": ["npm", "run", "lint", "--", target],
            "typecheck": ["npx", "tsc", "--noEmit"],
            "build": ["npm", "run", "build"],
        }
    elif ecosystem == "go":
        commands = {
            "test": ["go", "test", "./..."],
            "lint": ["go", "vet", "./..."],
            "build": ["go", "build", "./..."],
        }
    elif ecosystem == "rust":
        commands = {
            "test": ["cargo", "test"],
            "lint": ["cargo", "clippy", "--", "-D", "warnings"],
            "typecheck": ["cargo", "check"],
            "build": ["cargo", "build"],
        }
    elif ecosystem == "jvm":
        wrapper = "mvn" if (get_workspace_root() / "pom.xml").exists() else "gradle"
        commands = {
            "test": [wrapper, "test"],
            "build": [wrapper, "build"],
        }
    else:
        commands = {
            "test": ["make", "test"],
            "build": ["make", "build"],
        }
    return [(check, commands[check]) for check in checks if check in commands and check in selected]


def run_checks(
    ecosystem: str = "auto",
    checks: list[str] | None = None,
    target: str = ".",
    args: list[str] | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    """Run bounded test/build/lint checks for the detected project ecosystem."""
    ctx = get_sandbox_context()
    if ctx is None:
        return {"success": False, "error": "No active sandbox context", "checks": []}
    root = _workspace()
    detected, _ = _detect_ecosystems(root)
    chosen = ecosystem.lower().strip()
    if chosen == "auto":
        chosen = next((item for item in detected if item != "streamlit"), detected[0] if detected else "python")
    requested = checks or ["test"]
    commands = _commands_for(chosen, requested, target)
    if not commands:
        return {
            "success": False,
            "error": f"No supported checks for ecosystem '{chosen}' and requested checks {requested}.",
            "checks": [],
        }

    results: list[dict[str, Any]] = []
    for check, command in commands:
        argv = list(command) + list(args or [])
        result = get_executor().run_argv(
            argv,
            ctx=ctx,
            timeout=max(1, min(int(timeout), 600)),
            network=NetworkMode.NONE,
            profile="worker",
            cwd=root,
        )
        results.append({
            "check": check,
            "ecosystem": chosen,
            "argv": argv,
            "passed": result.get("returncode") == 0,
            "returncode": result.get("returncode", -1),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "timed_out": result.get("timed_out", False),
            "policy_denied": result.get("policy_denied", False),
        })
        if result.get("returncode") != 0:
            break

    return {
        "success": all(item["passed"] for item in results),
        "ecosystem": chosen,
        "checks": results,
    }
