"""
Session sandbox — filesystem jail, command policy, subprocess executors.

Phase 9 PR1+PR2: central gate for all subprocess execution.
"""

from src.sandbox.context import (
    SandboxContext,
    activate_sandbox,
    deactivate_sandbox,
    get_sandbox_context,
    sandbox_from_session,
)
from src.sandbox.commands import canonicalize_shell_script, compile_contract_to_argv, execute_contract
from src.sandbox.executor import SubprocessExecutor, get_executor, resolve_backend
from src.sandbox.policy import SandboxMode, describe_policy_for_profile, format_policy_reference, get_sandbox_mode, is_python_interpreter, validate_shell_script
from src.sandbox.probe import SandboxToolchainError, verify_sandbox_toolchain

__all__ = [
    "SandboxContext",
    "activate_sandbox",
    "deactivate_sandbox",
    "get_sandbox_context",
    "sandbox_from_session",
    "SubprocessExecutor",
    "get_executor",
    "resolve_backend",
    "canonicalize_shell_script",
    "compile_contract_to_argv",
    "execute_contract",
    "SandboxToolchainError",
    "verify_sandbox_toolchain",
    "SandboxMode",
    "describe_policy_for_profile",
    "format_policy_reference",
    "get_sandbox_mode",
    "is_python_interpreter",
    "validate_shell_script",
]
