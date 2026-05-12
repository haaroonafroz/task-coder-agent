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
# Increased for token-heavy pipeline to maximize speculative decoding benefit
# ---------------------------------------------------------------------------
MAX_TOKENS_PLANNER = 1500
MAX_TOKENS_GENERATOR = 2000
MAX_TOKENS_REFINER = 2000
MAX_TOKENS_DOC_GENERATOR = 1500


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

# Planner: Chain-of-Thought in prompt (not model reasoning) for token volume
_PLANNER_TEMPLATE = """\
You are a software architect. Produce a comprehensive implementation plan.

REQUIRED FORMAT:
1. **Problem Analysis** (2-3 paragraphs):
   - Break down what makes this problem complex
   - Identify constraints and edge cases
   - Discuss input/output characteristics

2. **Algorithm Evaluation** (evaluate 2-3 approaches):
   - Approach 1: Naive/brute force with pros/cons
   - Approach 2: Optimized solution with complexity analysis
   - Approach 3: Alternative strategy if applicable

3. **Edge Case Handling**:
   - List ALL edge cases (empty inputs, single elements, duplicates, etc.)
   - Explain how each will be handled

4. **Implementation Steps** (10-15 detailed steps):
   - Numbered list with specific variable names and data structures
   - Include helper functions needed
   - Detail validation and error checking

5. **Testing Strategy**:
   - What test cases cover normal operation
   - What tests cover edge cases
   - Expected time/space complexity

Be thorough and explicit.

Problem: {problem_text}

Implementation Plan:"""


# Generator: Verbose production-ready code for high token volume
_GENERATOR_TEMPLATE = """\
You are an expert Python programmer. Write a COMPLETE, PRODUCTION-READY solution.

REQUIREMENTS (all mandatory):
1. **Comprehensive Docstrings**: Every function must have Google-style docstrings with:
   - Detailed description of what the function does
   - Args: parameter names, types, and descriptions
   - Returns: return type and description
   - Raises: all possible exceptions
   - Time/space complexity analysis (Big O notation)

2. **Inline Comments**: Explain every non-trivial logic step

3. **Error Handling**: Full defensive programming with:
   - Type validation (isinstance checks)
   - Value validation (range checks, empty checks)
   - Custom exceptions with descriptive messages
   - try/except blocks where appropriate

4. **Type Hints**: Full PEP-484 type annotations for all functions

5. **Helper Functions**: Create separate functions for sub-tasks
   - Don't inline complex logic
   - Each helper must have its own docstring

6. **Example Usage Block**: Include if __name__ == "__main__": with 3+ examples

7. **Imports**: Include all necessary imports at the top

Rules:
- Output only raw Python code. No markdown. No backticks. No explanation text.
- The code must be complete and directly callable as a python function.

Problem: {problem_text}

Implementation Plan:
{planner_output}

Python code:{repeat_suffix}"""


# Refiner: Enterprise-grade code refinement for maximum token generation
_REFINER_TEMPLATE = """\
You are a senior Python code reviewer. Transform this into ENTERPRISE-GRADE code.

REQUIRED IMPROVEMENTS:
1. **Enhanced Docstrings**: If not already present, add comprehensive Google-style format:
   - Extended description with algorithm explanation
   - Detailed Args/Returns/Raises sections
   - 1-2 usage examples embedded in the docstring

2. **Defensive Programming**: Add comprehensive validation:
   - Precondition checks at function entry
   - Postcondition assertions where appropriate
   - Custom exception types

3. **Inline Documentation**: Comment every significant operation.

4. **Logging Integration**: Add optional logging support:
   - import logging with getLogger
   - Debug-level logs for key operations
   - Warning logs for edge cases handled

5. **Type Safety**: If not already present, add comprehensive type hints:
   - All function parameters and returns
   - Type variables (TypeVar) for generics
   - Union types where multiple types accepted
   - Optional[] for nullable parameters

6. **Code Organization**:
   - Constants as module-level UPPER_CASE
   - Private helpers prefixed with _
   - Clear separation of concerns

7. **Module-Level Documentation**: Add module docstring explaining:
   - Purpose of this module
   - Key functions overview

Output only the improved Python code directly executable as a python function. No explanations. No markdown. No backticks.

Code to improve:
{generated_code}

Improved code:{repeat_suffix}"""


# Doc Generator: Additional token-heavy node for comprehensive documentation
_DOC_GENERATOR_TEMPLATE = """\
You are a technical writer and API documentation specialist. Generate comprehensive external documentation for this code.

REQUIRED SECTIONS:
1. **Function API Reference** (for each public function):
   - Complete signature with all parameters
   - Detailed description of functionality
   - Parameter descriptions with valid ranges/types
   - Return value description
   - All exceptions that may be raised
   - Time and space complexity analysis

2. **Usage Examples** (minimum 5 examples):
   - Basic usage with simple inputs
   - Advanced usage with complex inputs
   - Edge case handling demonstrations
   - Error handling examples
   - Performance-critical usage patterns

3. **Implementation Notes**:
   - Algorithm explanation in plain English
   - Design decisions and trade-offs
   - Performance characteristics
   - Thread-safety considerations
   - Memory usage patterns

4. **Integration Guide**:
   - How to import and use in other projects
   - Dependencies and requirements
   - Configuration options if any

5. **Testing Recommendations**:
   - Unit test strategies
   - Integration test approaches
   - Property-based test ideas

Be thorough and verbose. Minimum 500 words. Use technical writing best practices.

Code to document:
{refined_code}

Documentation:{repeat_suffix}"""


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


def run_doc_generator(
    refined_code: str,
    mode: str,
    prompt_repetition: bool = True,
) -> LLMResult:
    """
    Generate comprehensive external documentation for the refined code.
    Adds significant token volume for better speculative decoding measurement.

    prompt_repetition: repeat the output-format constraint after the code
    context to counteract the lost-in-the-middle effect in small models.
    """
    suffix = _repeat(_REPEAT_REFINER, enabled=prompt_repetition)
    prompt = _DOC_GENERATOR_TEMPLATE.format(
        refined_code=refined_code,
        repeat_suffix=suffix,
    )
    return call_llm(prompt, mode=mode, max_tokens=MAX_TOKENS_DOC_GENERATOR)
