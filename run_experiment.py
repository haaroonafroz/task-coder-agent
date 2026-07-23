"""
Main experiment runner (Section 13).

Usage:
    # Calibration run (planner + generator only, baseline)
    python run_experiment.py --mode baseline --calibration

    # Full baseline run (all 50 tasks)
    python run_experiment.py --mode baseline

    # Full speculative run (all 50 tasks)
    python run_experiment.py --mode speculative

The script:
  1. Loads and MD5-verifies the frozen dataset
  2. Runs 3 warmup tasks (results discarded)
  3. Runs all 50 tasks through the full pipeline
  4. Appends per-task rows to experiments/results.csv
  5. Saves per-task outputs under experiments/outputs/
  6. Updates experiments/run_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.graph import run_task
from pipeline.agents import run_planner, run_generator, run_refiner  # for calibration mode
from pipeline.log_parser import update_results_with_acceptance_rates
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "datasets" / "mbpp_subset.json"
RESULTS_CSV = ROOT / "experiments" / "results.csv"
OUTPUTS_DIR = ROOT / "experiments" / "outputs"
METADATA_PATH = ROOT / "experiments" / "run_metadata.json"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CSV schema (Section 11.1)
# ---------------------------------------------------------------------------
CSV_FIELDNAMES = [
    "task_id", "mode",
    "planner_prefill_ms", "planner_decode_ms", "planner_total_ms", "planner_tokens_generated",
    "generator_prefill_ms", "generator_decode_ms", "generator_total_ms", "generator_tokens_generated",
    "refiner_prefill_ms", "refiner_decode_ms", "refiner_total_ms", "refiner_tokens_generated",
    "doc_generator_prefill_ms", "doc_generator_decode_ms", "doc_generator_total_ms", "doc_generator_tokens_generated",
    "pipeline_total_ms",
    "test_pass", "test_error_message",
    "acceptance_rate_generator",  # speculative mode only
]

# Number of warmup tasks
N_WARMUP = 3


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset() -> tuple[list[dict], str]:
    """Load the frozen MBPP subset and return (tasks, md5_hash)."""
    raw = DATASET_PATH.read_text()
    md5 = hashlib.md5(raw.encode()).hexdigest()
    tasks = json.loads(raw)
    return tasks, md5


def verify_dataset_md5(md5: str, expected: str | None) -> None:
    if expected and md5 != expected:
        print(f"[FATAL] Dataset MD5 mismatch! Expected {expected}, got {md5}.")
        print("        Do NOT proceed — the dataset has changed between runs.")
        sys.exit(1)
    print(f"[OK] Dataset MD5: {md5}")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def ensure_csv_header() -> None:
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()


def append_csv_row(row: dict) -> None:
    with open(RESULTS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Output file helpers (Section 11.2)
# ---------------------------------------------------------------------------

def save_task_outputs(task_id: int, mode: str, state: dict) -> None:
    prefix = OUTPUTS_DIR / f"task_{task_id}_{mode}"
    if "planner_text" in state:
        (prefix.parent / f"task_{task_id}_{mode}_plan.txt").write_text(
            state.get("planner_text", ""), encoding="utf-8"
        )
    if "generator_text" in state:
        (prefix.parent / f"task_{task_id}_{mode}_code.py").write_text(
            state.get("generator_text", ""), encoding="utf-8"
        )
    if "refiner_text" in state:
        (prefix.parent / f"task_{task_id}_{mode}_refined.py").write_text(
            state.get("refiner_text", ""), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Metadata helpers (Section 11.3)
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}


def save_metadata(meta: dict) -> None:
    METADATA_PATH.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Calibration run (planner + generator only, faster iteration)
# ---------------------------------------------------------------------------

def run_calibration(tasks: list[dict], mode: str) -> None:
    """
    Lightweight calibration: planner + generator + test runner only.
    Used to check E4B pass rate before committing to the full benchmark.
    """
    print(f"\n=== CALIBRATION RUN ({mode}) ===")
    print(f"Tasks: {len(tasks)} | Warmup: {N_WARMUP} | Agents: planner + generator only\n")

    passed = failed = 0

    for idx, task in enumerate(tqdm(tasks, desc="Calibration")):
        task_id = task["task_id"]
        is_warmup = idx < N_WARMUP
        label = "WARMUP" if is_warmup else f"task {task_id}"

        plan_result = run_planner(task["text"], mode=mode)
        gen_result = run_generator(task["text"], plan_result.text, mode=mode)

        from pipeline.test_runner import run_tests
        test_result = run_tests(gen_result.text, task["test_list"])

        if is_warmup:
            tqdm.write(f"  [{label}] (discarded)")
            continue

        status = "PASS" if test_result["passed"] else "FAIL"
        tqdm.write(f"  [{label}] {status}")
        if test_result["passed"]:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    rate = passed / total if total > 0 else 0.0
    print(f"\nCalibration complete: {passed}/{total} passed ({rate:.1%})")

    if rate < 0.30:
        print("[WARNING] Pass rate < 30%. Model may be too weak for this dataset.")
        print("          Consider switching to simpler tasks before running full benchmark.")
    elif rate < 0.60:
        print("[OK] Pass rate 30-60%. Proceed with awareness of model limitations.")
    else:
        print("[GREAT] Pass rate > 60%. Pipeline is functioning well.")


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def run_full_benchmark(tasks: list[dict], mode: str, dataset_md5: str) -> None:
    print(f"\n=== FULL BENCHMARK ({mode.upper()}) ===")
    print(f"Tasks: {len(tasks)} | Warmup: {N_WARMUP} | All agents\n")

    ensure_csv_header()

    meta = load_metadata()
    meta.setdefault("date", str(date.today()))
    meta.setdefault("target_model", "Qwen/Qwen3.6-35B-A3B")
meta.setdefault("draft_model", "native-mtp")  # Qwen has built-in MTP
meta.setdefault("num_speculative_tokens", 4)
    meta.setdefault("quantization", "bitsandbytes-8bit")
    meta.setdefault("temperature", 0)
    meta.setdefault("seed", 42)
    meta.setdefault("dataset", "mbpp_subset.json")
    meta["dataset_md5"] = dataset_md5
    meta.setdefault("warmup_tasks", N_WARMUP)
    meta.setdefault("total_tasks", len(tasks) - N_WARMUP)

    run_start = time.time()

    for idx, task in enumerate(tqdm(tasks, desc=f"[{mode}]")):
        task_id = task["task_id"]
        is_warmup = idx < N_WARMUP
        label = "WARMUP" if is_warmup else f"task {task_id}"

        try:
            state = run_task(
                task_id=task_id,
                problem_text=task["text"],
                test_list=task["test_list"],
                mode=mode,
                prompt_repetition=True,
            )
        except Exception as exc:
            tqdm.write(f"  [{label}] ERROR: {exc}")
            if is_warmup:
                continue
            # Log a failure row so the CSV stays consistent
            append_csv_row({
                "task_id": task_id,
                "mode": mode,
                "test_pass": False,
                "test_error_message": str(exc),
            })
            continue

        if is_warmup:
            tqdm.write(f"  [{label}] (warmup, discarded)")
            continue

        status = "PASS" if state.get("test_passed") else "FAIL"
        tqdm.write(
            f"  [task {task_id}] {status} | "
            f"pipeline={state.get('pipeline_total_ms', 0):.0f}ms | "
            f"gen_tokens={state.get('generator_tokens', 0)}"
        )

        row = {
            "task_id": task_id,
            "mode": mode,
            "planner_prefill_ms": state.get("planner_prefill_ms"),
            "planner_decode_ms": state.get("planner_decode_ms"),
            "planner_total_ms": state.get("planner_total_ms"),
            "planner_tokens_generated": state.get("planner_tokens"),
            "generator_prefill_ms": state.get("generator_prefill_ms"),
            "generator_decode_ms": state.get("generator_decode_ms"),
            "generator_total_ms": state.get("generator_total_ms"),
            "generator_tokens_generated": state.get("generator_tokens"),
            "refiner_prefill_ms": state.get("refiner_prefill_ms"),
            "refiner_decode_ms": state.get("refiner_decode_ms"),
            "refiner_total_ms": state.get("refiner_total_ms"),
            "refiner_tokens_generated": state.get("refiner_tokens"),
            "doc_generator_prefill_ms": state.get("doc_generator_prefill_ms"),
            "doc_generator_decode_ms": state.get("doc_generator_decode_ms"),
            "doc_generator_total_ms": state.get("doc_generator_total_ms"),
            "doc_generator_tokens_generated": state.get("doc_generator_tokens"),
            "pipeline_total_ms": state.get("pipeline_total_ms"),
            "test_pass": state.get("test_passed"),
            "test_error_message": state.get("test_stderr", ""),
            "acceptance_rate_generator": "",  # populated from vLLM logs post-hoc
        }
        append_csv_row(row)
        save_task_outputs(task_id, mode, state)

    elapsed = time.time() - run_start
    print(f"\nRun complete in {elapsed:.1f}s. Results appended to {RESULTS_CSV}")

    meta[f"run_complete_{mode}"] = True
    save_metadata(meta)

    if mode == "speculative":
        from pipeline.log_parser import update_results_with_acceptance_rates
        log_path = ROOT / "experiments" / "logs" / "speculative_server.log"
        update_results_with_acceptance_rates(RESULTS_CSV, log_path, mode="speculative")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speculative decoding evaluation – experiment runner"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "speculative"],
        required=True,
        help="Which vLLM server to target",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Run lightweight calibration (planner+generator only)",
    )
    parser.add_argument(
        "--expected-md5",
        default=None,
        help="Expected MD5 of mbpp_subset.json (cross-run integrity check)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="Limit to first N tasks (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--no-prompt-repetition",
        dest="prompt_repetition",
        action="store_false",
        default=True,
        help="Disable prompt repetition (off by default; use for ablation studies)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks, md5 = load_dataset()
    verify_dataset_md5(md5, args.expected_md5)

    if args.tasks:
        tasks = tasks[: args.tasks + N_WARMUP]
        print(f"[INFO] Limiting to {args.tasks} benchmark tasks + {N_WARMUP} warmup.")

    print(f"[INFO] Prompt repetition: {'ON' if args.prompt_repetition else 'OFF'}")

    if args.calibration:
        run_calibration(tasks, mode=args.mode)
    else:
        run_full_benchmark(tasks, mode=args.mode, dataset_md5=md5)


if __name__ == "__main__":
    main()
