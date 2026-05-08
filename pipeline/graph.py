"""
LangGraph pipeline for the speculative decoding evaluation.
Graph structure:
    START → load_task → planner → generator → refiner → test_runner → metrics → END
"""

from __future__ import annotations

import time
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from pipeline.agents import run_planner, run_generator, run_refiner
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

def planner_node(state: PipelineState) -> PipelineState:
    result = run_planner(state["problem_text"], mode=state["mode"])
    return {
        **state,
        "planner_text": result.text,
        "planner_prefill_ms": result.prefill_ms,
        "planner_decode_ms": result.decode_ms,
        "planner_total_ms": result.total_ms,
        "planner_tokens": result.tokens_generated,
    }


def generator_node(state: PipelineState) -> PipelineState:
    result = run_generator(
        problem_text=state["problem_text"],
        planner_output=state["planner_text"],
        mode=state["mode"],
        prompt_repetition=state.get("prompt_repetition", True),
    )
    return {
        **state,
        "generator_text": result.text,
        "generator_prefill_ms": result.prefill_ms,
        "generator_decode_ms": result.decode_ms,
        "generator_total_ms": result.total_ms,
        "generator_tokens": result.tokens_generated,
    }


def refiner_node(state: PipelineState) -> PipelineState:
    result = run_refiner(
        generated_code=state["generator_text"],
        mode=state["mode"],
        prompt_repetition=state.get("prompt_repetition", True),
    )
    return {
        **state,
        "refiner_text": result.text,
        "refiner_prefill_ms": result.prefill_ms,
        "refiner_decode_ms": result.decode_ms,
        "refiner_total_ms": result.total_ms,
        "refiner_tokens": result.tokens_generated,
    }


def test_runner_node(state: PipelineState) -> PipelineState:
    test_result = run_tests(
        code=state["refiner_text"],
        tests=state["test_list"],
    )
    return {
        **state,
        "test_passed": test_result["passed"],
        "test_stderr": test_result["stderr"],
        "test_stdout": test_result["stdout"],
        "test_exec_ms": test_result["exec_ms"],
    }


def metrics_node(state: PipelineState) -> PipelineState:
    """Compute pipeline_total_ms as sum of all agent total times."""
    pipeline_total = (
        state.get("planner_total_ms", 0.0)
        + state.get("generator_total_ms", 0.0)
        + state.get("refiner_total_ms", 0.0)
    )
    return {**state, "pipeline_total_ms": round(pipeline_total, 2)}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)

    builder.add_node("planner", planner_node)
    builder.add_node("generator", generator_node)
    builder.add_node("refiner", refiner_node)
    builder.add_node("test_runner", test_runner_node)
    builder.add_node("metrics", metrics_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "generator")
    builder.add_edge("generator", "refiner")
    builder.add_edge("refiner", "test_runner")
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
