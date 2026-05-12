"""
LangGraph pipeline for the speculative decoding evaluation.
Graph structure:
    START → load_task → planner → generator → refiner → test_runner → metrics → END
"""

from __future__ import annotations

import time
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from pipeline.agents import run_planner, run_generator, run_refiner, run_doc_generator
from pipeline.test_runner import run_tests


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    # Inputs
    mode: str                       # "baseline" or "speculative"
    task_id: int
    problem_text: str
    test_list: list[str]
    prompt_repetition: bool         # repeat core constraint at end of Generator/Refiner prompts

    # Planner outputs
    planner_text: str
    planner_prefill_ms: float
    planner_decode_ms: float
    planner_total_ms: float
    planner_tokens: int

    # Generator outputs
    generator_text: str
    generator_prefill_ms: float
    generator_decode_ms: float
    generator_total_ms: float
    generator_tokens: int

    # Refiner outputs
    refiner_text: str
    refiner_prefill_ms: float
    refiner_decode_ms: float
    refiner_total_ms: float
    refiner_tokens: int

    # Doc generator outputs
    doc_generator_text: str
    doc_generator_prefill_ms: float
    doc_generator_decode_ms: float
    doc_generator_total_ms: float
    doc_generator_tokens: int

    # Test runner outputs
    test_passed: bool
    test_stderr: str
    test_stdout: str
    test_exec_ms: float

    # Pipeline aggregate
    pipeline_start_ms: float
    pipeline_total_ms: float

    # Error flag
    error: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _log_node_entry(node_name: str, state: PipelineState) -> None:
    """Print clear marker when entering a node."""
    task_id = state.get("task_id", "?")
    print(f"\n{'='*60}")
    print(f"→ ENTERING NODE: {node_name} [task {task_id}]")
    print(f"{'='*60}")

def _log_node_exit(node_name: str, state: PipelineState, output_keys: list[str]) -> None:
    """Print clear marker when exiting a node with summary of outputs."""
    task_id = state.get("task_id", "?")
    print(f"\n{'='*60}")
    print(f"← EXITING NODE: {node_name} [task {task_id}]")
    print(f"  Passing to next node:")
    for key in output_keys:
        value = state.get(key)
        if value is not None:
            if isinstance(value, str):
                preview = value[:60].replace('\n', ' ')
                print(f"    • {key}: {preview}{'...' if len(value) > 60 else ''}")
            elif isinstance(value, (int, float)):
                print(f"    • {key}: {value}")
            else:
                print(f"    • {key}: {type(value).__name__}")
    print(f"{'='*60}")


def planner_node(state: PipelineState) -> PipelineState:
    _log_node_entry("planner", state)
    result = run_planner(state["problem_text"], mode=state["mode"])
    new_state: PipelineState = {
        **state,
        "planner_text": result.text,
        "planner_prefill_ms": result.prefill_ms,
        "planner_decode_ms": result.decode_ms,
        "planner_total_ms": result.total_ms,
        "planner_tokens": result.tokens_generated,
    }
    _log_node_exit("planner", new_state, ["planner_text", "planner_tokens", "planner_total_ms"])
    return new_state


def generator_node(state: PipelineState) -> PipelineState:
    _log_node_entry("generator", state)
    result = run_generator(
        problem_text=state["problem_text"],
        planner_output=state["planner_text"],
        mode=state["mode"],
        prompt_repetition=state.get("prompt_repetition", True),
    )
    new_state: PipelineState = {
        **state,
        "generator_text": result.text,
        "generator_prefill_ms": result.prefill_ms,
        "generator_decode_ms": result.decode_ms,
        "generator_total_ms": result.total_ms,
        "generator_tokens": result.tokens_generated,
    }
    _log_node_exit("generator", new_state, ["generator_text", "generator_tokens", "generator_total_ms"])
    return new_state



def refiner_node(state: PipelineState) -> PipelineState:
    _log_node_entry("refiner", state)
    result = run_refiner(
        generated_code=state["generator_text"],
        mode=state["mode"],
        prompt_repetition=state.get("prompt_repetition", True),
    )
    new_state: PipelineState = {
        **state,
        "refiner_text": result.text,
        "refiner_prefill_ms": result.prefill_ms,
        "refiner_decode_ms": result.decode_ms,
        "refiner_total_ms": result.total_ms,
        "refiner_tokens": result.tokens_generated,
    }
    _log_node_exit("refiner", new_state, ["refiner_text", "refiner_tokens", "refiner_total_ms"])
    return new_state


def doc_generator_node(state: PipelineState) -> PipelineState:
    _log_node_entry("doc_generator", state)
    result = run_doc_generator(
        refined_code=state["refiner_text"],
        mode=state["mode"],
        prompt_repetition=state.get("prompt_repetition", True),
    )
    new_state: PipelineState = {
        **state,
        "doc_generator_text": result.text,
        "doc_generator_prefill_ms": result.prefill_ms,
        "doc_generator_decode_ms": result.decode_ms,
        "doc_generator_total_ms": result.total_ms,
        "doc_generator_tokens": result.tokens_generated,
    }
    _log_node_exit("doc_generator", new_state, ["doc_generator_tokens", "doc_generator_total_ms"])
    return new_state


def test_runner_node(state: PipelineState) -> PipelineState:
    _log_node_entry("test_runner", state)
    test_result = run_tests(
        code=state["refiner_text"],
        tests=state["test_list"],
    )
    new_state: PipelineState = {
        **state,
        "test_passed": test_result["passed"],
        "test_stderr": test_result["stderr"],
        "test_stdout": test_result["stdout"],
        "test_exec_ms": test_result["exec_ms"],
    }
    _log_node_exit("test_runner", new_state, ["test_passed", "test_exec_ms"])
    return new_state



def metrics_node(state: PipelineState) -> PipelineState:
    _log_node_entry("metrics", state)
    pipeline_total = (
        state.get("planner_total_ms", 0.0)
        + state.get("generator_total_ms", 0.0)
        + state.get("refiner_total_ms", 0.0)
        + state.get("doc_generator_total_ms", 0.0)
    )
    new_state: PipelineState = {**state, "pipeline_total_ms": round(pipeline_total, 2)}
    _log_node_exit("metrics", new_state, ["pipeline_total_ms"])
    return new_state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)

    builder.add_node("planner", planner_node)
    builder.add_node("generator", generator_node)
    builder.add_node("refiner", refiner_node)
    builder.add_node("doc_generator", doc_generator_node)
    builder.add_node("test_runner", test_runner_node)
    builder.add_node("metrics", metrics_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "generator")
    builder.add_edge("generator", "refiner")
    builder.add_edge("refiner", "doc_generator")
    builder.add_edge("doc_generator", "test_runner")
    builder.add_edge("test_runner", "metrics")
    builder.add_edge("metrics", END)

    return builder.compile()


# Module-level compiled graph (import and use directly)
pipeline = build_graph()


def run_task(
    task_id: int,
    problem_text: str,
    test_list: list[str],
    mode: str,
    prompt_repetition: bool = True,
) -> PipelineState:
    """
    Run a single task through the full pipeline.
    Returns the final PipelineState with all timing and pass/fail fields populated.
    """
    initial_state: PipelineState = {
        "mode": mode,
        "task_id": task_id,
        "problem_text": problem_text,
        "test_list": test_list,
        "prompt_repetition": prompt_repetition,
        "error": None,
    }
    final_state: PipelineState = pipeline.invoke(initial_state)
    return final_state
