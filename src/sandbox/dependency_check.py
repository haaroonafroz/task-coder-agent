"""Detect and verify third-party Python dependencies for milestone target files."""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.sandbox.context import SandboxContext, get_sandbox_context
from src.sandbox.env import resolve_python
from src.sandbox.executor import get_executor
from src.tools.paths import get_workspace_root, normalize_workspace_path

# Pre-installed harness dev tools — not worker-installed mission deps.
_PREINSTALLED = frozenset({"pytest", "flake8", "black", "mypy", "ruff", "_pytest"})

# Common third-party libraries referenced in mission descriptions.
_THIRD_PARTY_HINT_RE = re.compile(
    r"\b("
    r"pygame|flask|django|fastapi|httpx|requests|numpy|pandas|pillow|sqlalchemy|"
    r"uvicorn|pydantic|matplotlib|scipy|torch|tensorflow|beautifulsoup4|bs4|"
    r"redis|celery|boto3|aiohttp|websockets|click|typer|jinja2|yaml|"
    r"email_validator|phonenumbers|cryptography|jwt|passlib"
    r")\b",
    re.IGNORECASE,
)

# Import name → PyPI package name when they differ.
_IMPORT_TO_PACKAGE: dict[str, str] = {
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "jwt": "PyJWT",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
}


@dataclass
class DependencyReport:
    """Result of scanning target files and checking the session venv."""

    required_imports: list[str] = field(default_factory=list)
    missing_imports: list[str] = field(default_factory=list)
    missing_packages: list[str] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_packages and not self.errors


def _stdlib_top_level() -> frozenset[str]:
    names = getattr(sys, "stdlib_module_names", None)
    if names:
        return frozenset(names)
    return frozenset()


def _resolve_target_path(rel: str, workspace_root: Path) -> Path:
    p = Path(normalize_workspace_path(rel))
    if p.is_absolute():
        return p
    resolved = (workspace_root / p).resolve()
    root_resolved = workspace_root.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise ValueError(f"Path escapes workspace: {rel}")
    return resolved


def _local_module_names(workspace_root: Path) -> frozenset[str]:
    """Top-level module/package names defined in the workspace."""
    names: set[str] = set()
    if not workspace_root.is_dir():
        return frozenset(names)

    for path in workspace_root.rglob("*"):
        if path.name.startswith(".") or "__pycache__" in path.parts:
            continue
        if path.is_dir() and (path / "__init__.py").exists():
            try:
                rel = path.relative_to(workspace_root)
                if rel.parts:
                    names.add(rel.parts[0])
            except ValueError:
                continue
        elif path.suffix == ".py" and path.name != "__init__.py":
            try:
                rel = path.relative_to(workspace_root)
                names.add(rel.stem)
                if rel.parts:
                    names.add(rel.parts[0])
            except ValueError:
                continue
    return frozenset(names)


def _collect_import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _package_name_for_import(import_root: str) -> str:
    mapped = _IMPORT_TO_PACKAGE.get(import_root)
    if mapped:
        return mapped
    return import_root.replace("_", "-").lower() if import_root.islower() else import_root


def planned_module_names(plan: Optional[dict]) -> frozenset[str]:
    """Top-level module names the mission plan will create locally.

    TDD test-scaffold milestones import the module under test BEFORE it
    exists (red phase). Deriving planned module names from the plan's target
    files prevents misclassifying those imports as missing third-party
    packages — which previously rejected the worker's COMPLETE and pushed it
    to ``pip install`` nonexistent packages.
    """
    names: set[str] = set()
    if not plan:
        return frozenset()
    for ms in plan.get("milestones", []) or []:
        if not isinstance(ms, dict):
            continue
        for rel in ms.get("target_files", []) or []:
            p = Path(str(rel))
            if p.suffix == ".py" and p.name != "__init__.py":
                names.add(p.stem)
                if len(p.parts) > 1:
                    names.add(p.parts[0])
    return frozenset(names)


def collect_third_party_imports(
    file_paths: list[str],
    *,
    workspace_root: Optional[Path] = None,
    planned_modules: Optional[frozenset[str]] = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Parse target files and return (required_import_roots, checked_files, errors).
    """
    ws = (workspace_root or get_workspace_root()).resolve()
    stdlib = _stdlib_top_level()
    local = _local_module_names(ws) | (planned_modules or frozenset())
    required: set[str] = set()
    checked: list[str] = []
    errors: list[str] = []

    for rel in file_paths:
        rel = rel.strip()
        if not rel:
            continue
        try:
            path = _resolve_target_path(rel, ws)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.exists() or path.suffix != ".py":
            continue
        checked.append(rel)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as exc:
            errors.append(f"{rel}: syntax error at line {exc.lineno}: {exc.msg}")
            continue
        for root in _collect_import_roots(tree):
            if root in stdlib or root in local or root in _PREINSTALLED:
                continue
            required.add(root)

    return sorted(required), checked, errors


def third_party_hints_in_text(*texts: str) -> list[str]:
    """Return deduplicated third-party package hints found in free text."""
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in _THIRD_PARTY_HINT_RE.finditer(text):
            token = match.group(1)
            pkg = _package_name_for_import(token)
            key = pkg.lower()
            if key not in seen:
                seen.add(key)
                found.append(pkg)
    return found


def is_import_installed(import_root: str, *, ctx: Optional[SandboxContext] = None) -> bool:
    """Return True when ``import_root`` resolves in the active session venv."""
    sandbox = ctx or get_sandbox_context()
    if sandbox is None:
        return False

    python = resolve_python(sandbox)
    snippet = (
        "import importlib.util, sys; "
        f"sys.exit(0 if importlib.util.find_spec({import_root!r}) else 1)"
    )
    result = get_executor().run_argv(
        [python, "-c", snippet],
        ctx=sandbox,
        timeout=30,
        profile="validation",
    )
    return result.get("returncode") == 0


def check_target_file_dependencies(
    file_paths: list[str],
    *,
    ctx: Optional[SandboxContext] = None,
    workspace_root: Optional[Path] = None,
    planned_modules: Optional[frozenset[str]] = None,
) -> DependencyReport:
    """
    Scan ``file_paths`` for third-party imports and verify they are installed.
    """
    sandbox = ctx or get_sandbox_context()
    ws = workspace_root
    if ws is None and sandbox is not None:
        ws = sandbox.workspace_root
    if ws is None:
        ws = get_workspace_root()

    required, checked, errors = collect_third_party_imports(
        file_paths,
        workspace_root=ws,
        planned_modules=planned_modules,
    )
    missing_imports: list[str] = []
    missing_packages: list[str] = []

    for import_root in required:
        if is_import_installed(import_root, ctx=ctx):
            continue
        missing_imports.append(import_root)
        pkg = _package_name_for_import(import_root)
        if pkg not in missing_packages:
            missing_packages.append(pkg)

    return DependencyReport(
        required_imports=required,
        missing_imports=missing_imports,
        missing_packages=missing_packages,
        checked_files=checked,
        errors=errors,
    )


def format_missing_dependency_message(report: DependencyReport) -> str:
    """Human-readable guidance for worker/validator failures."""
    parts: list[str] = []
    if report.missing_packages:
        pkg_list = ", ".join(report.missing_packages)
        parts.append(
            f"Missing third-party packages in the session venv: {pkg_list}. "
            f"Call install_dependency for each package before signalling complete."
        )
    if report.errors:
        parts.append("Dependency scan errors: " + "; ".join(report.errors))
    return " ".join(parts) if parts else "Missing third-party dependencies."


def milestone_suggests_dependencies(milestone: dict, plan: Optional[dict] = None) -> bool:
    """True when milestone/plan text hints that third-party packages are required."""
    texts = [
        milestone.get("title", ""),
        milestone.get("description", ""),
    ]
    if plan:
        texts.extend([
            plan.get("title", ""),
            plan.get("description", ""),
        ])
    return bool(third_party_hints_in_text(*texts))
