from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
WORKSPACE_ROOT = _REPO_ROOT / "workspace"

def normalize_workspace_path(path: str) -> str:
    """Strip workspace/ prefix; return path relative to workspace root."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if p.startswith("workspace/"):
        p = p[len("workspace/"):]
    return p.lstrip("/")

def resolve_workspace_path(path: str) -> Path:
    """Resolve a workspace-relative path to an absolute Path under workspace/."""
    p = Path(normalize_workspace_path(path))
    if p.is_absolute():
        return p
    resolved = (WORKSPACE_ROOT / p).resolve()
    # Safety: block path escape
    if WORKSPACE_ROOT.resolve() not in resolved.parents and resolved != WORKSPACE_ROOT.resolve():
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved

def normalize_shell_command(command: str) -> str:
    """Rewrite workspace/foo → foo for commands run with cwd=workspace/."""
    return command.replace("workspace/", "")