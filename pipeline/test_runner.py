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
import re

def extract_code(text: str) -> str:
    """
    Strip markdown fences, thinking blocks, and reasoning tokens from LLM output.
    Handles Gemma 4's thinking structure including truncated thinking.
    """
    # 0. Check for complete Gemma 4 thinking block
    # Format: <|channel1>thought\n [reasoning] <|channel1|> [Final answer]
    complete_thinking = r"<\|channel1>thought\n.*?<\|channel1\|>\s*(.*)"
    match = re.search(complete_thinking, text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    
    # 1. Check for "Final Answer" markers (common in reasoning models)
    final_answer_markers = [
        r"(?:Final\s+Answer|Final\s+Code|Answer\s*:)\\s*\\n?",
        r"(?:Here\s+is\s+the\s+(?:improved\s+)?(?:Python\s+)?code\s*:?)\\s*\\n?",
        r"(?:Output\s*(?:only\s*)?(?:the\s*)?(?:improved\s+)?code\s*:?)\\s*\\n?",
    ]
    for marker in final_answer_markers:
        match = re.search(marker + r"(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            break
    
    # 2. Try fenced code block
    match = re.search(r"```(?:python)?\\s*\\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 3. Handle truncated thinking - look for code-start patterns from bottom up
    lines = text.strip().splitlines()
    
    # Patterns that indicate code (not reasoning)
    code_start_patterns = [
        r"^def\\s+\\w+",           # function definition
        r"^class\\s+\\w+",         # class definition  
        r"^import\\s+\\w+",       # import statement
        r"^from\\s+\\w+",         # from import
        r"^@\\w+",                 # decorator
        r"^#\\s*\\w+.*:",          # comment with colon (often docstring style)
    ]
    
    # Patterns that indicate reasoning (not code)
    reasoning_patterns = [
        r"^(?:Wait|Let\\s+me|I\\s+(?:need|will|should|think|can)|Let's|Okay|So|Hmm)",
        r"^(?:Self-Correction|Re-evaluating|Wait,|Actually,|No,)",
        r"^(?:The\\s+user\\s+wants|I\\s+need\\s+to|I\\s+should)",
        r"^(?:One\\s+(?:more|final)|Last\\s+check|Final\\s+check)",
        r"^[*-]\\s",               # bullet points
        r"^\\d+\\.",               # numbered lists
    ]
    
    # Find the last line that starts with code pattern
    code_start_idx = None
    for i in range(len(lines)):
        line = lines[i].lstrip()
        if not line:
            continue
            
        is_code = any(re.match(p, line) for p in code_start_patterns)
        is_reasoning = any(re.match(p, line, re.IGNORECASE) for p in reasoning_patterns)
        
        if is_code and not is_reasoning:
            code_start_idx = i
    
    if code_start_idx is not None:
        code_end_idx = len(lines)
        for i in range(code_start_idx + 1, len(lines)):
            line = lines[i].lstrip()
            if not line:
                continue
            if any(re.match(p, line, re.IGNORECASE) for p in reasoning_patterns):
                if not line.startswith("#"):
                    code_end_idx = i
                    break
        
        return "\\n".join(lines[code_start_idx:code_end_idx]).strip()
    
    # 4. Ultimate fallback - find first code-like line
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if any(stripped.startswith(kw) for kw in ["def ", "class ", "import ", "from ", "@"]):
            return "\\n".join(lines[i:]).strip()
    
    return text.strip()

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
    code = extract_code(code)
    test_block = "\n".join(tests)
    full_script = (
        "import resource as _resource\n"
        "_resource.setrlimit(_resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))\n"
        "\n"
        f"{code}\n"
        "\n"
        f"{test_block}\n"
        'print("PASS")\n'
    )

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
