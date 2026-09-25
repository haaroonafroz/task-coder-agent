"""Context-efficiency guards: prompt-tree caps, arg aliases, venv resolution.

Regression: a project .venv with ~40k files inflated a hotfix prompt to
304k tokens and killed a run against a 128k window. The workspace tree must
stay small no matter what lives in the project root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.tools.file_ops as file_ops
from src.sandbox.context import SandboxContext
from src.sandbox.env import resolve_python
from src.tools.file_ops import list_directory
from src.tools.paths import reset_workspace_root, set_workspace_root
from src.tools.tool_contracts import normalize_tool_args, validate_tool_call


@pytest.fixture()
def ws(tmp_path: Path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    set_workspace_root(root)
    try:
        yield root
    finally:
        reset_workspace_root()


def _big_dir(root: Path, name: str, files: int = 300) -> None:
    target = root / name
    target.mkdir(exist_ok=True)
    for i in range(files):
        (target / f"mod_{i}.py").write_text("x = 1\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# tree exclusions + cap
# ---------------------------------------------------------------------------

def test_tree_hides_venv_but_names_it(ws: Path) -> None:
    _big_dir(ws, ".venv")
    result = list_directory(".", 6)
    assert result["success"] is True
    assert ".venv/ (contents hidden)" in result["tree"]
    assert "mod_0.py" not in result["tree"]
    assert result["count"] < 20


def test_tree_skips_caches_and_build_dirs(ws: Path) -> None:
    for name in ("__pycache__", "node_modules", ".git", "wandb", "dist"):
        _big_dir(ws, name, files=50)
    result = list_directory(".", 6)
    assert "mod_0.py" not in result["tree"]
    for name in ("__pycache__", "node_modules", ".git", "wandb", "dist"):
        assert f"{name}/ (contents hidden)" in result["tree"]


def test_tree_truncates_huge_projects(ws: Path, monkeypatch) -> None:
    _big_dir(ws, "data", files=500)
    monkeypatch.setattr(file_ops, "_MAX_TREE_CHARS", 500)
    result = list_directory(".", 6)
    assert result["truncated"] is True
    assert len(result["tree"]) < 1000
    assert "truncated" in result["tree"]


def test_tree_small_project_untouched(ws: Path) -> None:
    result = list_directory(".", 6)
    assert result["success"] is True
    assert result["truncated"] is False
    assert "src/app.py" in result["tree"] or "app.py" in result["tree"]


# ---------------------------------------------------------------------------
# arg aliases
# ---------------------------------------------------------------------------

def test_aliases_rewrite_observed_mistakes() -> None:
    assert normalize_tool_args(
        "list_directory", {"path": "src"}
    ) == {"target_dir": "src"}
    assert normalize_tool_args(
        "read_file", {"file_path": "a.py", "start_line": 10}
    ) == {"file_path": "a.py", "offset": 10}
    assert normalize_tool_args(
        "search_grep", {"pattern": "foo", "path": "."}
    ) == {"query": "foo", "target_dir": "."}
    assert normalize_tool_args(
        "run_shellscript", {"command": "ls"}
    ) == {"script": "ls"}


def test_explicit_args_win_over_aliases() -> None:
    out = normalize_tool_args(
        "read_file", {"file_path": "a.py", "offset": 5, "start_line": 10}
    )
    assert out["offset"] == 5
    assert "start_line" in out  # left for the validator to reject
    assert validate_tool_call("read_file", out) is not None


def test_aliased_calls_validate_clean() -> None:
    args = normalize_tool_args("list_directory", {"path": "."})
    assert validate_tool_call("list_directory", args) is None
    args = normalize_tool_args(
        "read_file", {"path": "a.py", "start_line": 1}
    )
    assert validate_tool_call("read_file", args) is None


def test_normalize_passthrough() -> None:
    assert normalize_tool_args("unknown_tool", {"a": 1}) == {"a": 1}
    assert normalize_tool_args("read_file", "nope") == "nope"


# ---------------------------------------------------------------------------
# resolve_python
# ---------------------------------------------------------------------------

def _ctx(tmp_path: Path, kind: str, mode: str, venv: Path) -> SandboxContext:
    state = tmp_path / "state"
    return SandboxContext(
        session_id="s1",
        jail_root=state,
        workspace_root=tmp_path / "proj",
        venv_path=venv,
        tmp_dir=state / ".tmp",
        home_dir=state / ".home",
        pip_cache_dir=state / ".cache" / "pip",
        workspace_kind=kind,
        workspace_mode=mode,
    )


def test_resolve_prefers_existing_project_venv(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    python = proj / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    ctx = _ctx(tmp_path, "external", "read_write", tmp_path / "state" / ".venv")
    assert resolve_python(ctx) == str(python)


def test_resolve_read_only_never_provisions(tmp_path: Path) -> None:
    (tmp_path / "proj").mkdir(exist_ok=True)
    ctx = _ctx(tmp_path, "external", "read_only", tmp_path / "state" / ".venv")
    import sys

    assert resolve_python(ctx) == sys.executable
    assert not (tmp_path / "proj" / ".venv").exists()


def test_resolve_managed_provisions_session_venv(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "proj").mkdir(exist_ok=True)
    ctx = _ctx(tmp_path, "managed", "read_write", tmp_path / "state" / ".venv")
    monkeypatch.setattr(
        SandboxContext, "ensure_venv", lambda self: Path("/session/python")
    )
    assert resolve_python(ctx) == "/session/python"


# ---------------------------------------------------------------------------
# worker↔validator retry loop: dedupe, repeat breaker, tail-heavy diff
# ---------------------------------------------------------------------------

from src.agents.validator import _bounded_workspace_diff
from src.agents.worker import _drop_redundant_constraints, _retry_entry_message
from src.main import _normalize_failure_signature


class _Mem:
    """Minimal memory stub exposing get_error_constraints()."""

    def __init__(self, block: str) -> None:
        self._block = block

    def get_error_constraints(self, top_n: int = 5) -> str:  # noqa: ARG002
        return self._block


def _milestone() -> dict:
    return {"id": "M1", "title": "t", "target_files": ["src/app.py"]}


def test_retry_drops_constraint_duplicating_failure_report() -> None:
    err = "AssertionError: expected 42 but got 7 in compute_total for input batch nine"
    feedback = (
        f"[Retry 1] Errors: ['{err}'] | Guidance: fix the aggregation logic "
        f"\nRaw validation output (bounded):\n{err}"
    )
    constraints = (
        "## Negative Constraints (do NOT repeat these patterns)\n"
        f"- File src/app.py failed validation with error: {err}. "
        "Do not replicate this syntax or structure."
    )
    msg = _retry_entry_message(feedback, _Mem(constraints), _milestone(), ["src/app.py"])
    assert "Validator failure report" in msg
    # Same error text must not appear twice.
    assert "Do not replicate this syntax" not in msg


def test_retry_keeps_novel_constraints_from_older_attempts() -> None:
    feedback = "[Retry 2] Errors: ['ValueError: bad shape (3, 4)'] | Guidance: reshape"
    old_err = "TypeError: unsupported operand for CauseNumber in the legacy parser path"
    constraints = (
        "## Negative Constraints (do NOT repeat these patterns)\n"
        f"- File src/app.py failed validation with error: {old_err}. "
        "Do not replicate this syntax or structure."
    )
    msg = _retry_entry_message(feedback, _Mem(constraints), _milestone(), ["src/app.py"])
    assert old_err in msg


def test_retry_without_feedback_keeps_constraints() -> None:
    constraints = (
        "## Negative Constraints (do NOT repeat these patterns)\n"
        "- File src/app.py failed validation with error: boom. "
        "Do not replicate this syntax or structure."
    )
    msg = _retry_entry_message(None, _Mem(constraints), _milestone(), ["src/app.py"])
    assert "boom" in msg


def test_failure_signature_ignores_line_numbers_but_not_error_kind() -> None:
    a = _normalize_failure_signature(["AssertionError: expected 1 at line 12 in test_x"])
    b = _normalize_failure_signature(["AssertionError: expected 1 at line 34 in test_x"])
    c = _normalize_failure_signature(["TypeError: unsupported operand at line 12"])
    assert a == b
    assert a != c
    assert _normalize_failure_signature([]) == ""


def test_diff_truncation_is_tail_heavy(monkeypatch) -> None:
    import src.agents.validator as validator_mod

    diff = "\n".join(f"line-{i:04d}-padding-to-grow-the-diff-body" for i in range(600))
    monkeypatch.setattr(validator_mod, "git_diff", lambda: {"success": True, "diff": diff})
    block = _bounded_workspace_diff()
    assert "... [diff truncated" in block
    assert "line-0599" in block  # the fix attempt at the tail survives
    assert "line-0000" in block  # small head kept for context
    assert "line-0300" not in block  # middle truncated away
