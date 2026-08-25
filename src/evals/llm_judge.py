"""LLM-as-judge evaluators (harness-native; optional via MISSIONS_EVAL_LLM_JUDGE)."""

from __future__ import annotations

import json
import os
from typing import Any

from src.agents.utils import parse_json_from_text
from src.evals.types import EvalContext, EvalScore
from src.llm_client import ModelChoice, call_llm

_PASS_THRESHOLD = 0.7


def _parse_judge_response(text: str, name: str) -> EvalScore:
    parsed = parse_json_from_text(text)
    if not parsed or "score" not in parsed:
        return EvalScore(
            name=name,
            score=0.0,
            passed=False,
            summary="Judge returned unparseable output",
            details={"raw": text[:500]},
            kind="llm_judge",
        )
    try:
        score = float(parsed["score"])
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    return EvalScore(
        name=name,
        score=round(score, 4),
        passed=score >= _PASS_THRESHOLD,
        summary=str(parsed.get("summary", ""))[:300],
        details={
            "issues": parsed.get("issues", []),
            "rationale": parsed.get("rationale", ""),
        },
        kind="llm_judge",
    )


def eval_plan_quality(ctx: EvalContext, model: ModelChoice = "auto") -> EvalScore:
    """LLM judge: is the milestone decomposition sensible for the user request?"""
    if not ctx.plan.get("milestones"):
        return EvalScore(
            name="plan_quality",
            score=0.0,
            passed=False,
            summary="No plan milestones to evaluate",
            kind="llm_judge",
        )

    prompt = (
        "You are an expert software project evaluator.\n"
        "Score the mission plan quality from 0.0 to 1.0.\n\n"
        f"User request:\n{ctx.user_request}\n\n"
        f"Plan JSON:\n{json.dumps(ctx.plan, indent=2)[:6000]}\n\n"
        "Return JSON only:\n"
        '{"score": 0.0, "summary": "...", "rationale": "...", "issues": ["..."]}'
    )
    result = call_llm(prompt, model=model, max_tokens=1024, json_mode=True, role="validator")
    return _parse_judge_response(result.text, "plan_quality")


def eval_contract_quality(ctx: EvalContext, model: ModelChoice = "auto") -> EvalScore:
    """LLM judge: are validation contracts meaningful and testable?"""
    milestones = ctx.plan.get("milestones", [])
    contracts = [
        {
            "id": m.get("id"),
            "title": m.get("title"),
            "validation_contract": m.get("validation_contract", {}),
            "target_files": m.get("target_files", []),
        }
        for m in milestones
    ]
    if not contracts:
        return EvalScore(
            name="contract_quality",
            score=0.0,
            passed=False,
            summary="No validation contracts to evaluate",
            kind="llm_judge",
        )

    prompt = (
        "You are an expert test/validation reviewer.\n"
        "Score validation contract quality from 0.0 to 1.0.\n"
        "Contracts should be runnable, specific, and aligned with deliverables.\n\n"
        f"Contracts:\n{json.dumps(contracts, indent=2)[:6000]}\n\n"
        "Return JSON only:\n"
        '{"score": 0.0, "summary": "...", "rationale": "...", "issues": ["..."]}'
    )
    result = call_llm(prompt, model=model, max_tokens=1024, json_mode=True, role="validator")
    return _parse_judge_response(result.text, "contract_quality")


def eval_trajectory_review(ctx: EvalContext, model: ModelChoice = "auto") -> EvalScore:
    """LLM judge: review condensed event trajectory for wasted work / gaming risk."""
    # Condense events to keep prompt bounded
    condensed: list[dict[str, Any]] = []
    for ev in ctx.events:
        condensed.append({
            "ts": ev.get("ts"),
            "type": ev.get("type"),
            "data": {
                k: v for k, v in ev.get("data", {}).items()
                if k in (
                    "milestone_id", "verdict", "tool", "status",
                    "retry", "reason", "errors", "unauthorized_edits",
                )
            },
        })
    prompt = (
        "You are an expert agentic harness evaluator.\n"
        "Review the mission event trajectory and score overall execution quality "
        "from 0.0 to 1.0 (tool discipline, retry waste, spec-gaming risk).\n\n"
        f"Events ({len(condensed)}):\n{json.dumps(condensed[-80:], indent=2)[:8000]}\n\n"
        "Return JSON only:\n"
        '{"score": 0.0, "summary": "...", "rationale": "...", "issues": ["..."]}'
    )
    result = call_llm(prompt, model=model, max_tokens=1024, json_mode=True, role="validator")
    return _parse_judge_response(result.text, "trajectory_review")


def run_llm_judge_evals(
    ctx: EvalContext,
    model: ModelChoice = "auto",
) -> list[EvalScore]:
    """Run all LLM-as-judge evaluators."""
    return [
        eval_plan_quality(ctx, model=model),
        eval_contract_quality(ctx, model=model),
        eval_trajectory_review(ctx, model=model),
    ]


def llm_judges_enabled() -> bool:
    return os.getenv("MISSIONS_EVAL_LLM_JUDGE", "false").strip().lower() == "true"
