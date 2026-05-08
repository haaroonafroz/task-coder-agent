"""
Agent definitions: Planner, Code Generator, Refiner.

All prompt templates are frozen here.

PROMPT REPETITION (Prompt Repetition Improves Non-Reasoning LLMs):
The repeated suffix is a compressed restatement of the key output constraint.
"""

from __future__ import annotations

from pipeline.llm_client import LLMResult, call_llm

# ---------------------------------------------------------------------------
# Token budget constants (Section 7.3)
# ---------------------------------------------------------------------------
MAX_TOKENS_PLANNER = 250
MAX_TOKENS_GENERATOR = 700
MAX_TOKENS_REFINER = 700


# ---------------------------------------------------------------------------
# Prompt repetition helper
# ---------------------------------------------------------------------------

def _repeat(suffix: str, enabled: bool = True) -> str:
    """
    Return the repeated-instruction suffix when enabled.

    Setting enabled=False lets us A/B the technique independently of
    the speculative decoding comparison (useful for ablation studies).
    """
    if not enabled:
        return ""
    return f"\n\nReminder: {suffix}"


# Core constraints — one compact sentence per agent, echoed at the end.
_REPEAT_GENERATOR = (
    "Output ONLY raw Python code. No markdown, no backticks, no explanation."
)
_REPEAT_REFINER = (
    "Output ONLY the improved Python code. No markdown, no explanations."
)


# ---------------------------------------------------------------------------
# Frozen prompt templates (Section 5) with prompt repetition applied
# ---------------------------------------------------------------------------

# Planner: no injected context → repetition not needed
_PLANNER_TEMPLATE = """\
You are a software architect. Given the following coding problem,
produce a concise implementation plan.

Format your response as a numbered list of steps.
Do not write code. Describe logic and structure only.
Keep your response under 200 words.

Problem: {problem_text}

Implementation Plan:"""


# Generator: context = plan (can be 80–180 tokens) → repetition applied
_GENERATOR_TEMPLATE = """\
You are an expert Python programmer.
Using the implementation plan below, write a complete Python function
that solves the problem.

Rules:
- Output only raw Python code. No markdown. No backticks. No explanation.
- Include the function signature and all helper functions.
- The function must be self-contained and importable.

Problem: {problem_text}

Implementation Plan:
{planner_output}

Python code:{repeat_suffix}"""


# Refiner: context = generated code (200–500 tokens) → repetition applied
_REFINER_TEMPLATE = """\
You are a Python code reviewer. Improve the following code by:
1. Adding PEP-484 type hints to all function signatures
2. Improving variable names if unclear
3. Adding a one-line docstring
4. Simplifying any unnecessarily complex logic

Output only the improved Python code. No explanations. No markdown.

Code to improve:
{generated_code}

Improved code:{repeat_suffix}"""


# ---------------------------------------------------------------------------
# Agent callables
# ---------------------------------------------------------------------------

def run_planner(problem_text: str, mode: str) -> LLMResult:
    """Produce a structured implementation plan for the given problem."""
    prompt = _PLANNER_TEMPLATE.format(problem_text=problem_text)
    return call_llm(prompt, mode=mode, max_tokens=MAX_TOKENS_PLANNER)


def run_generator(
    problem_text: str,
    planner_output: str,
    mode: str,
    prompt_repetition: bool = True,
) -> LLMResult:
    """
    Generate a complete Python function from the plan.

    prompt_repetition: repeat the output-format constraint after the plan
    context to counteract the lost-in-the-middle effect in small models.
    """
    suffix = _repeat(_REPEAT_GENERATOR, enabled=prompt_repetition)
    prompt = _GENERATOR_TEMPLATE.format(
        problem_text=problem_text,
        planner_output=planner_output,
        repeat_suffix=suffix,
    )
    return call_llm(prompt, mode=mode, max_tokens=MAX_TOKENS_GENERATOR)


def run_refiner(
    generated_code: str,
    mode: str,
    prompt_repetition: bool = True,
) -> LLMResult:
    """
    Improve the generated code with type hints, naming, and docstrings.

    prompt_repetition: repeat the output-format constraint after the code
    context to counteract the lost-in-the-middle effect in small models.
    """
    suffix = _repeat(_REPEAT_REFINER, enabled=prompt_repetition)
    prompt = _REFINER_TEMPLATE.format(
        generated_code=generated_code,
        repeat_suffix=suffix,
    )
    return call_llm(prompt, mode=mode, max_tokens=MAX_TOKENS_REFINER)
