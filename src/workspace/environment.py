"""Detect existing project environments without creating dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.workspace.models import ProjectEnvironment


def detect_environment(root: Path) -> Optional[ProjectEnvironment]:
    for name in (".venv", "venv"):
        env_root = root / name
        python = env_root / "bin" / "python"
        if python.is_file():
            return ProjectEnvironment(kind="python-venv", root=env_root, python=python)

    if (root / "uv.lock").is_file():
        return ProjectEnvironment(kind="python-manifest", package_manager="uv")
    if (root / "poetry.lock").is_file():
        return ProjectEnvironment(kind="python-manifest", package_manager="poetry")
    if (root / "Pipfile").is_file():
        return ProjectEnvironment(kind="python-manifest", package_manager="pipenv")
    if (root / "pyproject.toml").is_file() or (root / "requirements.txt").is_file():
        return ProjectEnvironment(kind="python-manifest", package_manager="pip")

    if (root / "node_modules").is_dir():
        manager = _node_package_manager(root)
        return ProjectEnvironment(
            kind="node-modules", root=root / "node_modules", package_manager=manager
        )
    if (root / "package.json").is_file():
        return ProjectEnvironment(kind="node-manifest", package_manager=_node_package_manager(root))

    if (root / "Cargo.toml").is_file():
        return ProjectEnvironment(kind="rust", package_manager="cargo")
    if (root / "go.mod").is_file():
        return ProjectEnvironment(kind="go", package_manager="go")
    return None


def _node_package_manager(root: Path) -> str:
    if (root / "pnpm-lock.yaml").is_file():
        return "pnpm"
    if (root / "yarn.lock").is_file():
        return "yarn"
    return "npm"
