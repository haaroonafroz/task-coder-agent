"""
Command policy engine for sandboxed shell execution.

Profiles:
  worker     — balanced allowlist for implementation commands
  validation — stricter profile for validator validation_contract commands
  pip        — only pip install/uninstall via install_dependency tool

Modes (SANDBOX_MODE env):
  strict   — allowlist only
  balanced — blocklist dangerous ops + allowlist (default)
  permissive — blocklist only (dev)
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

ShellProfile = str  # "worker" | "validation" | "pip" | "devserver" | "browser"


class SandboxMode(str, Enum):
    STRICT = "strict"
    BALANCED = "balanced"
    PERMISSIVE = "permissive"


class NetworkMode(str, Enum):
    NONE = "none"
    PIP_EGRESS = "pip"  # allow network for pip only
    LOCALHOST = "localhost"  # reserved for harness-managed local services


# Blocked in all profiles.
_GLOBAL_BLOCK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bsudo\b",
        r"\bsu\s",
        r"\bcurl\b",
        r"\bwget\b",
        r"\bssh\b",
        r"\bscp\b",
        r"\bnc\b",
        r"\btelnet\b",
        r"\bkill\b",
        r"\bpkill\b",
        r"\bkillall\b",
        r"\bchmod\b",
        r"\bchown\b",
        r"\bmount\b",
        r"\bumount\b",
        r"\bdd\b",
        r"\bmkfs\b",
        r"rm\s+-rf\s+/\s*",
        r"rm\s+-rf\s+/\*",
        r">\s*/etc/",
        r">\s*~/",
        r">\s*\$HOME",
        r"\$\(\s*curl",
        r"`curl",
        r"&&\s*curl",
        r"\|\s*bash",
        r"\|\s*sh\s",
        r"&\s*$",           # background jobs
        r";\s*&",
    ]
]

# Shell-builtin patterns — worker only (validation may reference eval() in tests).
_WORKER_ONLY_BLOCK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\beval\b",
        r"\bexec\b",
    ]
]

# Balanced / strict allowlist — first token of each command segment.
_WORKER_ALLOW: set[str] = {
    "pytest", "python", "python3", "flake8", "black", "mypy", "ruff",
    "ls", "find", "cat", "head", "tail", "wc", "grep", "rg", "echo",
    "test", "[", "true", "false", "pwd", "dirname", "basename", "sort",
    "uniq", "diff", "make", "touch", "mkdir", "cp", "mv", "rm",
    "cd", "env", "which", "file", "stat", "tree",
}

_VALIDATION_ALLOW: set[str] = {
    "pytest", "python", "python3", "flake8", "black", "mypy", "ruff",
    "test", "[", "true", "false", "echo", "ls", "cat", "pwd",
}

_PIP_ALLOW: set[str] = {"python", "python3"}
_DEVSERVER_ALLOW: set[str] = {
    "python", "python3", "node", "npm", "pnpm", "yarn", "npx",
    "streamlit", "uvicorn", "vite",
}
_BROWSER_ALLOW: set[str] = {"python", "python3", "node", "npx"}

_ALLOWED_PY_MODULES_WORKER: set[str] = {
    "pytest", "flake8", "black", "mypy", "ruff", "compileall", "py_compile",
}

_ALLOWED_PY_MODULES_VALIDATION: set[str] = {
    "pytest", "flake8", "black", "mypy", "ruff", "py_compile", "compileall",
}

_ALLOWED_PY_MODULES_PIP: set[str] = {"pip"}

_PYTHON_INTERPRETER_SUFFIXES = ("/bin/python", "/bin/python3")


@dataclass
class PolicyVerdict:
    allowed: bool
    reason: str = ""


def get_sandbox_mode() -> SandboxMode:
    raw = os.getenv("SANDBOX_MODE", "balanced").strip().lower()
    try:
        return SandboxMode(raw)
    except ValueError:
        return SandboxMode.BALANCED


def _split_segments(script: str) -> list[str]:
    """Split a shell script into segments on ; && || and newlines."""
    parts = re.split(r"(?:;|&&|\|\||\n)", script)
    return [p.strip() for p in parts if p.strip()]


def _first_command_token(segment: str) -> str:
    """Extract the first command token, skipping leading env assignments."""
    segment = segment.strip()
    while re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", segment):
        segment = re.sub(r"^[A-Za-z_][A-Za-z0-9_]*=", "", segment, count=1).strip()
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()[0] if segment.split() else ""
    return tokens[0] if tokens else ""


def is_python_interpreter(token: str) -> bool:
    """True for bare python/python3 or an absolute session venv interpreter path."""
    if token in ("python", "python3"):
        return True
    return any(token.endswith(suffix) for suffix in _PYTHON_INTERPRETER_SUFFIXES)


def _check_blocklist(script: str, profile: ShellProfile) -> Optional[str]:
    for pat in _GLOBAL_BLOCK_PATTERNS:
        if pat.search(script):
            return f"Blocked pattern matched: {pat.pattern}"

    if profile == "worker":
        for pat in _WORKER_ONLY_BLOCK_PATTERNS:
            if pat.search(script):
                return f"Blocked pattern matched: {pat.pattern}"

    if re.search(r"(?<![\w/])(/etc/|/root/|/home/[^/]+/\.ssh)", script):
        return "Blocked: absolute path outside session jail"
    return None


def _allowlist_for_profile(profile: ShellProfile) -> set[str]:
    if profile == "validation":
        return _VALIDATION_ALLOW
    if profile == "pip":
        return _PIP_ALLOW
    if profile == "devserver":
        return _DEVSERVER_ALLOW
    if profile == "browser":
        return _BROWSER_ALLOW
    return _WORKER_ALLOW


def _py_modules_for_profile(profile: ShellProfile) -> set[str]:
    if profile == "validation":
        return _ALLOWED_PY_MODULES_VALIDATION
    if profile == "pip":
        return _ALLOWED_PY_MODULES_PIP
    if profile in {"devserver", "browser"}:
        return {
            "streamlit", "uvicorn", "http.server", "http.server",
        }
    return _ALLOWED_PY_MODULES_WORKER


def _check_python_module(segment: str, profile: ShellProfile) -> bool:
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        return False
    if len(tokens) < 3:
        return False
    if not is_python_interpreter(tokens[0]):
        return False
    if tokens[1] != "-m":
        return False
    return tokens[2] in _py_modules_for_profile(profile)


def _segment_allowed(segment: str, profile: ShellProfile, mode: SandboxMode) -> PolicyVerdict:
    block = _check_blocklist(segment, profile)
    if block:
        return PolicyVerdict(False, block)

    token = _first_command_token(segment)
    if not token:
        return PolicyVerdict(False, "Empty command segment")

    if mode == SandboxMode.PERMISSIVE:
        return PolicyVerdict(True)

    allow = _allowlist_for_profile(profile)

    if token in allow or is_python_interpreter(token):
        if token == "rm" and profile != "pip":
            if re.search(r"\s+(-rf?|--recursive)\s+/", segment):
                return PolicyVerdict(False, "rm with absolute path blocked")
        # Block `python -m pip` outside the dedicated pip profile
        try:
            tokens = shlex.split(segment, posix=True)
            if (
                len(tokens) >= 3
                and tokens[1] == "-m"
                and tokens[2] == "pip"
                and profile != "pip"
            ):
                return PolicyVerdict(
                    False,
                    f"pip module not allowed in {profile} shell profile",
                )
        except ValueError:
            pass
        return PolicyVerdict(True)

    if is_python_interpreter(token) and _check_python_module(segment, profile):
        return PolicyVerdict(True)

    if mode == SandboxMode.BALANCED and profile == "worker":
        if segment.strip().startswith("pytest"):
            return PolicyVerdict(True)

    return PolicyVerdict(
        False,
        f"Command '{token}' not in {profile} allowlist (mode={mode.value})",
    )


def describe_policy_for_profile(profile: ShellProfile) -> dict[str, Any]:
    """
    Human- and LLM-readable summary of allowed commands for a sandbox profile.

    Included in validator REPLAN guidance when a contract is policy-denied.
    """
    py_modules = sorted(_py_modules_for_profile(profile))
    shell_cmds = sorted(_allowlist_for_profile(profile))
    examples = [
        "python -m pytest tests/test_x.py --collect-only -q",
        "python -m pytest tests/test_x.py -v -k tokenizer",
        "python -m flake8 module.py --max-line-length=120",
        "python -m py_compile tests/test_x.py",
    ]
    if profile == "validation":
        examples = [e for e in examples if "pip" not in e]

    return {
        "profile": profile,
        "shell_commands": shell_cmds,
        "python_modules": py_modules,
        "interpreter_tokens": ["python", "python3", "<session>/.venv/bin/python"],
        "recommended_contracts": examples,
        "notes": [
            "Harness rewrites bare python/python3 to the session venv interpreter.",
            "Prefer `python -m pytest ...` over custom shell pipelines.",
            "Test-scaffolding milestones: use --collect-only, not grep/py_compile tricks.",
            "Do not use pip install, curl, sudo, or bash eval/exec in contracts.",
        ],
    }


def format_policy_reference(profile: ShellProfile) -> str:
    """Format policy reference as markdown for validator/orchestrator prompts."""
    ref = describe_policy_for_profile(profile)
    lines = [
        f"**Profile**: `{ref['profile']}`",
        f"**Allowed shell commands**: {', '.join(f'`{c}`' for c in ref['shell_commands'])}",
        f"**Allowed `python -m` modules**: {', '.join(f'`{m}`' for m in ref['python_modules'])}",
        "**Recommended contract examples**:",
    ]
    for ex in ref["recommended_contracts"]:
        lines.append(f"  - `{ex}`")
    lines.append("**Notes**:")
    for note in ref["notes"]:
        lines.append(f"  - {note}")
    return "\n".join(lines)


def validate_shell_script(
    script: str,
    profile: ShellProfile = "worker",
    mode: Optional[SandboxMode] = None,
) -> PolicyVerdict:
    """
    Validate a shell script against the policy for the given profile.

    Returns PolicyVerdict(allowed=True) or PolicyVerdict(allowed=False, reason=...).
    """
    if mode is None:
        mode = get_sandbox_mode()

    script = script.strip()
    if not script:
        return PolicyVerdict(False, "Empty script")

    block = _check_blocklist(script, profile)
    if block:
        return PolicyVerdict(False, block)

    segments = _split_segments(script)
    for seg in segments:
        verdict = _segment_allowed(seg, profile, mode)
        if not verdict.allowed:
            return verdict

    return PolicyVerdict(True)


def validate_argv(
    argv: list[str],
    profile: ShellProfile = "worker",
) -> PolicyVerdict:
    """Validate an argv list (no shell) with profile-aware restrictions."""
    if not argv:
        return PolicyVerdict(False, "Empty argv")
    block = _check_blocklist(" ".join(argv), profile)
    if block:
        return PolicyVerdict(False, block)
    if profile in {"devserver", "browser"}:
        token = _first_command_token(" ".join(argv))
        allow = _allowlist_for_profile(profile)
        if token not in allow and not is_python_interpreter(token):
            return PolicyVerdict(
                False,
                f"Command '{token}' not in {profile} allowlist",
            )
    return PolicyVerdict(True)
