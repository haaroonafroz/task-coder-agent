"""Deterministic, bounded inspection for managed and external workspaces."""

from __future__ import annotations

from pathlib import Path

from src.workspace.environment import detect_environment
from src.workspace.git_state import snapshot_git_state
from src.workspace.models import ProjectProfile, WorkspaceKind

_MANIFESTS = {
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "Pipfile": "python",
    "package.json": "javascript",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "java",
    "Makefile": "generic",
}

_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "target", "dist",
    "__pycache__", ".pytest_cache",
}


def inspect_workspace(root: Path, kind: WorkspaceKind) -> ProjectProfile:
    root = root.resolve()
    manifests: list[str] = []
    languages: set[str] = set()
    for name, language in _MANIFESTS.items():
        if (root / name).exists():
            manifests.append(name)
            languages.add(language)

    scanned = 0
    for path in root.rglob("*"):
        if scanned >= 1000:
            break
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        scanned += 1
        language = _EXTENSIONS.get(path.suffix.lower())
        if language:
            languages.add(language)

    commands: dict[str, str] = {}
    if "python" in languages:
        commands["test"] = "python -m pytest"
    if "package.json" in manifests:
        commands.update(test="npm test", build="npm run build", lint="npm run lint")
    if "go" in languages:
        commands.update(test="go test ./...", build="go build ./...")
    if "rust" in languages:
        commands.update(test="cargo test", build="cargo build", lint="cargo clippy")

    return ProjectProfile(
        root=root,
        workspace_kind=kind,
        git=snapshot_git_state(root),
        languages=sorted(languages),
        manifests=sorted(manifests),
        environment=detect_environment(root),
        detected_commands=commands,
    )
