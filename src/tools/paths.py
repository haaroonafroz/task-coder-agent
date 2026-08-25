from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent

# Default workspace root (legacy). The active workspace root is mutable so
# that session-scoped runs can point all file/shell tools at
# sessions/<id>/workspace/ without changing call sites.
_DEFAULT_WORKSPACE_ROOT = _REPO_ROOT / "workspace"
_workspace_root: Path = _DEFAULT_WORKSPACE_ROOT


def set_workspace_root(path: Path) -> None:
    """Set the active workspace root for all subsequent tool calls.

    Called by the runtime before each session run. Because execution is
    serial, only one workspace root is active at a time.
    """
    global _workspace_root
    _workspace_root = Path(path)


def get_workspace_root() -> Path:
    """Return the currently active workspace root."""
    return _workspace_root


def reset_workspace_root() -> None:
    """Restore the legacy default workspace root (repo/workspace/)."""
    global _workspace_root
    _workspace_root = _DEFAULT_WORKSPACE_ROOT


# Backwards-compatible alias. Importing this name captures the *value* at
# import time, so new code should prefer get_workspace_root(). Kept so that
# external scripts and notebooks that did `from src.tools.paths import
# WORKSPACE_ROOT` keep working against the default location.
WORKSPACE_ROOT = _DEFAULT_WORKSPACE_ROOT


def normalize_workspace_path(path: str) -> str:
    """Strip workspace/ prefix; return path relative to workspace root."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if p.startswith("workspace/"):
        p = p[len("workspace/"):]
    return p.lstrip("/")


def resolve_workspace_path(path: str) -> Path:
    """Resolve a workspace-relative path to an absolute Path under the active workspace root."""
    p = Path(normalize_workspace_path(path))
    if p.is_absolute():
        return p
    root = get_workspace_root()
    resolved = (root / p).resolve()
    # Safety: block path escape
    root_resolved = root.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def normalize_shell_command(command: str) -> str:
    """Rewrite workspace/foo → foo for commands run with cwd=<workspace root>."""
    return command.replace("workspace/", "")
