"""
Subprocess executors for sandboxed command execution.

Backends (SANDBOX_EXECUTOR env):
  auto   — bwrap if available, else native (default)
  bwrap  — bubblewrap with jail mounts + --unshare-net
  native — policy-checked subprocess with process groups
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.sandbox.context import SandboxContext, get_sandbox_context
from src.sandbox.env import build_sandbox_env, resolve_python
from src.sandbox.policy import (
    NetworkMode,
    ShellProfile,
    validate_argv,
    validate_shell_script,
)
from src.settings import get_settings

ExecResult = dict[str, Any]


class ExecutorBackend(str, Enum):
    AUTO = "auto"
    NATIVE = "native"
    BWRAP = "bwrap"


@dataclass
class RunSpec:
    """Specification for one sandboxed execution."""

    cwd: Path
    env: dict[str, str]
    timeout: int
    network: NetworkMode = NetworkMode.NONE
    profile: ShellProfile = "worker"
    jail_root: Optional[Path] = None
    workspace_mode: str = "read_write"
    sandbox_required: bool = False


def _bwrap_available() -> bool:
    return shutil.which("bwrap") is not None


def resolve_backend() -> ExecutorBackend:
    raw = get_settings().runtime.sandbox.executor
    if raw == "bwrap":
        return ExecutorBackend.BWRAP
    if raw == "native":
        return ExecutorBackend.NATIVE
    return ExecutorBackend.BWRAP if _bwrap_available() else ExecutorBackend.NATIVE


def _bwrap_required() -> bool:
    return bool(get_settings().runtime.sandbox.require_bwrap)


def _deny_missing_bwrap(
    sandbox: SandboxContext,
    backend: ExecutorBackend,
) -> Optional[ExecResult]:
    """Fail closed only when an external jail is required and bwrap is mandatory."""
    using_bwrap = backend == ExecutorBackend.BWRAP and _bwrap_available()
    if sandbox.sandbox_required and not using_bwrap and _bwrap_required():
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Sandbox required for external workspace, but bubblewrap is unavailable",
            "success": False,
            "timed_out": False,
            "sandbox_denied": True,
        }
    return None


def _kill_process_group(proc: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _collect_result(
    proc: subprocess.Popen,
    timeout: int,
    cwd: Path,
) -> ExecResult:
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "returncode": proc.returncode if proc.returncode is not None else -1,
            "stdout": (stdout or "").strip() if isinstance(stdout, str) else (stdout or b"").decode(errors="replace").strip(),
            "stderr": (stderr or "").strip() if isinstance(stderr, str) else (stderr or b"").decode(errors="replace").strip(),
            "success": proc.returncode == 0,
            "timed_out": False,
            "cwd": str(cwd),
        }
    except subprocess.TimeoutExpired:
        _kill_process_group(proc, signal.SIGTERM)
        try:
            proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc, signal.SIGKILL)
            try:
                proc.communicate(timeout=1)
            except Exception:
                pass
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Process timed out after {timeout}s",
            "success": False,
            "timed_out": True,
            "cwd": str(cwd),
        }


def _native_popen(
    cmd: list[str] | str,
    *,
    shell: bool,
    spec: RunSpec,
) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        shell=shell,
        cwd=str(spec.cwd),
        env=spec.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        executable="/bin/bash" if shell else None,
        preexec_fn=os.setsid,
    )


_BWRAP_ENV_KEYS = (
    "PATH", "VIRTUAL_ENV", "MISSIONS_PYTHON", "PYTHONPATH",
    "HOME", "TMPDIR", "PIP_CACHE_DIR", "LANG", "LC_ALL", "NO_COLOR", "TERM",
)


def _bwrap_cmd(inner_cmd: list[str], spec: RunSpec) -> list[str]:
    """Build a bwrap invocation with independent state and workspace mounts."""
    state = (spec.jail_root or spec.cwd).resolve()
    workspace = spec.cwd.resolve()

    cmd: list[str] = ["bwrap"]

    # Minimal OS filesystem (read-only)
    for mount in ("/usr", "/bin", "/lib", "/lib64", "/etc/resolv.conf", "/etc/ssl", "/etc/pki"):
        p = Path(mount)
        if p.exists():
            cmd.extend(["--ro-bind", str(p), str(p)])

    # Harness state is always writable. The project mount follows the binding
    # access mode and may live anywhere outside the harness state tree.
    cmd.extend(["--bind", str(state), str(state)])
    if workspace != state:
        mount_flag = "--ro-bind" if spec.workspace_mode == "read_only" else "--bind"
        cmd.extend([mount_flag, str(workspace), str(workspace)])

    # A harness/system Python may live outside both roots (for example a
    # deployment venv or Conda prefix). Mount only its environment prefix RO.
    python_raw = spec.env.get("MISSIONS_PYTHON")
    if python_raw:
        python_path = Path(python_raw).resolve()
        covered = (
            python_path == state
            or state in python_path.parents
            or python_path == workspace
            or workspace in python_path.parents
            or any(
                base == python_path or base in python_path.parents
                for base in (Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"))
            )
        )
        if not covered:
            prefix = python_path.parent.parent if python_path.parent.name == "bin" else python_path.parent
            cmd.extend(["--ro-bind", str(prefix), str(prefix)])
    cmd.extend(["--chdir", str(workspace)])

    # Explicit env inside the jail (do not rely on shell inheritance)
    for key in _BWRAP_ENV_KEYS:
        value = spec.env.get(key)
        if value is not None:
            cmd.extend(["--setenv", key, value])

    # Process / dev
    cmd.extend(["--proc", "/proc", "--dev", "/dev"])

    # Security flags
    cmd.extend([
        "--die-with-parent",
        "--new-session",
    ])
    if spec.network == NetworkMode.NONE:
        cmd.append("--unshare-net")

    cmd.append("--")
    cmd.extend(inner_cmd)
    return cmd


class SubprocessExecutor:
    """Unified sandbox executor — selects native or bwrap backend."""

    def __init__(self, backend: Optional[ExecutorBackend] = None) -> None:
        self.backend = backend or resolve_backend()

    def _annotate_result(
        self,
        result: ExecResult,
        *,
        spec: RunSpec,
        ctx: SandboxContext,
    ) -> ExecResult:
        """Attach harness execution metadata for debugging and parity checks."""
        result["python"] = spec.env.get("MISSIONS_PYTHON") or resolve_python(ctx)
        result["venv"] = str(ctx.venv_path)
        result["profile"] = spec.profile
        result["executor"] = self.backend.value
        result["cwd"] = str(spec.cwd)
        return result

    def _spec_from_ctx(
        self,
        ctx: SandboxContext,
        *,
        timeout: int,
        network: NetworkMode = NetworkMode.NONE,
        profile: ShellProfile = "worker",
        use_venv: bool = True,
        cwd: Optional[Path] = None,
        env_overlay: Optional[dict[str, str]] = None,
    ) -> RunSpec:
        env = build_sandbox_env(ctx, use_venv=use_venv)
        if env_overlay:
            env.update({str(k): str(v) for k, v in env_overlay.items()})
        return RunSpec(
            cwd=cwd or ctx.workspace_root,
            env=env,
            timeout=timeout,
            network=network,
            profile=profile,
            jail_root=ctx.jail_root,
            workspace_mode=ctx.workspace_mode,
            sandbox_required=ctx.sandbox_required,
        )

    def run_argv(
        self,
        argv: list[str],
        *,
        ctx: Optional[SandboxContext] = None,
        timeout: int = 120,
        network: NetworkMode = NetworkMode.NONE,
        profile: ShellProfile = "worker",
        cwd: Optional[Path] = None,
        use_venv: bool = True,
        env_overlay: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute argv (no shell) inside the sandbox."""
        sandbox = ctx or get_sandbox_context()
        if sandbox is None:
            return {
                "returncode": -1, "stdout": "", "stderr": "No active sandbox context",
                "success": False, "timed_out": False,
            }

        denied = _deny_missing_bwrap(sandbox, self.backend)
        if denied is not None:
            return denied

        verdict = validate_argv(argv, profile=profile)
        if not verdict.allowed:
            from src.sandbox.env import resolve_python
            return {
                "returncode": -1, "stdout": "", "stderr": f"Policy denied: {verdict.reason}",
                "success": False, "timed_out": False, "policy_denied": True,
                "python": resolve_python(sandbox),
            }

        spec = self._spec_from_ctx(
            sandbox, timeout=timeout, network=network,
            profile=profile, use_venv=use_venv, cwd=cwd,
            env_overlay=env_overlay,
        )

        if self.backend == ExecutorBackend.BWRAP and _bwrap_available():
            bwrap_argv = _bwrap_cmd(argv, spec)
            proc = _native_popen(bwrap_argv, shell=False, spec=spec)
            result = _collect_result(proc, timeout, spec.cwd)
            if (
                result["returncode"] != 0
                and "bwrap:" in result.get("stderr", "")
                and not spec.sandbox_required
            ):
                proc = _native_popen(argv, shell=False, spec=spec)
                result = _collect_result(proc, timeout, spec.cwd)
                result["executor_fallback"] = "native"
            elif result["returncode"] != 0 and "bwrap:" in result.get("stderr", ""):
                result["sandbox_denied"] = True
            return self._annotate_result(result, spec=spec, ctx=sandbox)
        else:
            proc = _native_popen(argv, shell=False, spec=spec)

        return self._annotate_result(
            _collect_result(proc, timeout, spec.cwd),
            spec=spec,
            ctx=sandbox,
        )

    def run_shell(
        self,
        script: str,
        *,
        ctx: Optional[SandboxContext] = None,
        timeout: int = 30,
        network: NetworkMode = NetworkMode.NONE,
        profile: ShellProfile = "worker",
        env_overlay: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute a shell script inside the sandbox after policy check."""
        sandbox = ctx or get_sandbox_context()
        if sandbox is None:
            return {
                "returncode": -1, "stdout": "", "stderr": "No active sandbox context",
                "success": False, "timed_out": False,
            }

        denied = _deny_missing_bwrap(sandbox, self.backend)
        if denied is not None:
            return denied

        verdict = validate_shell_script(script, profile=profile)
        if not verdict.allowed:
            from src.sandbox.env import resolve_python
            return {
                "returncode": -1, "stdout": "", "stderr": f"Policy denied: {verdict.reason}",
                "success": False, "timed_out": False, "policy_denied": True,
                "cwd": str(sandbox.workspace_root),
                "python": resolve_python(sandbox),
            }

        spec = self._spec_from_ctx(
            sandbox,
            timeout=timeout,
            network=network,
            profile=profile,
            env_overlay=env_overlay,
        )

        if self.backend == ExecutorBackend.BWRAP and _bwrap_available():
            inner = ["/bin/bash", "-c", script]
            bwrap_argv = _bwrap_cmd(inner, spec)
            proc = _native_popen(bwrap_argv, shell=False, spec=spec)
            result = _collect_result(proc, timeout, spec.cwd)
            if (
                result["returncode"] != 0
                and "bwrap:" in result.get("stderr", "")
                and not spec.sandbox_required
            ):
                proc = _native_popen(script, shell=True, spec=spec)
                result = _collect_result(proc, timeout, spec.cwd)
                result["executor_fallback"] = "native"
            elif result["returncode"] != 0 and "bwrap:" in result.get("stderr", ""):
                result["sandbox_denied"] = True
        else:
            proc = _native_popen(script, shell=True, spec=spec)
            result = _collect_result(proc, timeout, spec.cwd)

        return self._annotate_result(result, spec=spec, ctx=sandbox)


# Module-level singleton
_executor: Optional[SubprocessExecutor] = None


def get_executor() -> SubprocessExecutor:
    global _executor
    if _executor is None:
        _executor = SubprocessExecutor()
    return _executor
