"""
Sandboxed test runner (Section 5.4).

Executes LLM-generated code in a subprocess with:
  - Hard 10-second wall-clock timeout
  - 512 MB memory cap via resource.RLIMIT_AS
  - Temporary file isolated to /tmp

Returns a dict with keys: pass, stderr, stdout, exec_ms
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
import time
from typing import TypedDict


class TestResult(TypedDict):
    passed: bool
    stderr: str
    stdout: str
    exec_ms: float


def run_tests(code: str, tests: list[str], timeout: int = 10) -> TestResult:
    """
    Run the given code string against a list of assert statements.

    The test block is appended to the code and executed in a subprocess.
    'PASS' is printed at the end of the script; its presence in stdout
    confirms all assertions succeeded.
    """
    test_block = "\n".join(tests)
    full_script = textwrap.dedent(f"""\
        import resource as _resource
        _resource.setrlimit(_resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        {code}
        {test_block}
        print("PASS")
    """)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, dir="/tmp"
        ) as fh:
            fh.write(full_script)
            tmp_path = fh.name

        t_start = time.perf_counter()
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        exec_ms = (time.perf_counter() - t_start) * 1000.0

        passed = result.returncode == 0 and "PASS" in result.stdout
        return TestResult(
            passed=passed,
            stderr=result.stderr.strip(),
            stdout=result.stdout.strip(),
            exec_ms=round(exec_ms, 2),
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            passed=False,
            stderr="TIMEOUT",
            stdout="",
            exec_ms=float(timeout * 1000),
        )
    except Exception as exc:
        return TestResult(
            passed=False,
            stderr=str(exc),
            stdout="",
            exec_ms=0.0,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
