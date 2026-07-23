"""Unit tests for sandbox command policy."""

from __future__ import annotations

from src.sandbox.policy import (
    SandboxMode,
    describe_policy_for_profile,
    is_python_interpreter,
    validate_shell_script,
)


def test_blocks_curl():
    v = validate_shell_script("curl https://evil.com", profile="worker")
    assert not v.allowed


def test_blocks_sudo():
    v = validate_shell_script("sudo rm -rf /", profile="worker")
    assert not v.allowed


def test_allows_pytest_balanced():
    v = validate_shell_script("pytest tests/ -v", profile="worker", mode=SandboxMode.BALANCED)
    assert v.allowed


def test_allows_python_m_pytest_validation():
    v = validate_shell_script(
        "python -m pytest tests/test_x.py -q",
        profile="validation",
        mode=SandboxMode.BALANCED,
    )
    assert v.allowed


def test_allows_canonicalized_venv_python_path():
    venv_python = "/home/user/sessions/abc123/.venv/bin/python"
    v = validate_shell_script(
        f"{venv_python} -m pytest tests/test_math_eval.py --collect-only -q",
        profile="validation",
        mode=SandboxMode.BALANCED,
    )
    assert v.allowed


def test_allows_python_m_py_compile_validation():
    v = validate_shell_script(
        "python -m py_compile tests/test_x.py",
        profile="validation",
        mode=SandboxMode.BALANCED,
    )
    assert v.allowed


def test_validation_allows_eval_in_test_path_not_grep():
    """Validation profile must not block math_eval.py paths or python -c checks."""
    v = validate_shell_script(
        "python -m py_compile tests/test_math_eval.py",
        profile="validation",
        mode=SandboxMode.BALANCED,
    )
    assert v.allowed


def test_worker_blocks_eval_in_grep():
    v = validate_shell_script(
        "grep -q 'assert.*eval(' tests/test_math_eval.py",
        profile="worker",
        mode=SandboxMode.BALANCED,
    )
    assert not v.allowed
    assert "eval" in v.reason.lower()


def test_validation_blocks_grep():
    v = validate_shell_script("grep -q foo tests/test_x.py", profile="validation")
    assert not v.allowed


def test_validation_blocks_pip():
    v = validate_shell_script("pip install requests", profile="validation")
    assert not v.allowed


def test_worker_blocks_pip_in_shell():
    v = validate_shell_script("python -m pip install requests", profile="worker")
    # pip module not in worker allowlist for shell
    assert not v.allowed


def test_blocks_rm_rf_root():
    v = validate_shell_script("rm -rf /", profile="worker")
    assert not v.allowed


def test_is_python_interpreter():
    assert is_python_interpreter("python")
    assert is_python_interpreter("python3")
    assert is_python_interpreter("/sessions/x/.venv/bin/python")
    assert not is_python_interpreter("grep")


def test_describe_policy_for_validation():
    ref = describe_policy_for_profile("validation")
    assert "pytest" in ref["shell_commands"]
    assert "pytest" in ref["python_modules"]
    assert "py_compile" in ref["python_modules"]
    assert any("collect-only" in ex for ex in ref["recommended_contracts"])
