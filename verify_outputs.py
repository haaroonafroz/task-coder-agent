"""
Output identity verification (Section 8).

At temperature=0 with a fixed seed, speculative decoding MUST produce
byte-for-byte identical outputs to baseline decoding.  Any mismatch
indicates either a correctness bug in the MTP implementation or a
non-zero effective temperature.

Usage:
    python verify_outputs.py

The script reads all refined output files under experiments/outputs/,
compares baseline vs speculative for every task, and writes the result
to experiments/run_metadata.json under the key "output_identity_verified".

Exit codes:
    0 – all outputs identical (safe to publish)
    1 – mismatches found     (do NOT publish without investigation)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUTS_DIR = ROOT / "experiments" / "outputs"
METADATA_PATH = ROOT / "experiments" / "run_metadata.json"

# Which output file to compare: refined code is the end product
OUTPUT_SUFFIX = "_refined.py"


def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}


def save_metadata(meta: dict) -> None:
    METADATA_PATH.write_text(json.dumps(meta, indent=2))


def collect_task_ids() -> list[int]:
    """Find all task IDs that have both baseline and speculative refined outputs."""
    baseline_ids: set[int] = set()
    spec_ids: set[int] = set()

    for p in OUTPUTS_DIR.glob("task_*_baseline_refined.py"):
        try:
            task_id = int(p.name.split("_")[1])
            baseline_ids.add(task_id)
        except (IndexError, ValueError):
            pass

    for p in OUTPUTS_DIR.glob("task_*_speculative_refined.py"):
        try:
            task_id = int(p.name.split("_")[1])
            spec_ids.add(task_id)
        except (IndexError, ValueError):
            pass

    common = sorted(baseline_ids & spec_ids)
    only_baseline = baseline_ids - spec_ids
    only_spec = spec_ids - baseline_ids

    if only_baseline:
        print(f"[WARN] Tasks with baseline output but no speculative: {sorted(only_baseline)}")
    if only_spec:
        print(f"[WARN] Tasks with speculative output but no baseline: {sorted(only_spec)}")

    return common


def verify_output_identity() -> bool:
    """
    Compare baseline vs speculative refined outputs for every task.
    Returns True if all outputs are identical, False otherwise.
    """
    task_ids = collect_task_ids()

    if not task_ids:
        print("[ERROR] No paired task outputs found in experiments/outputs/.")
        print("        Run both baseline and speculative experiments first.")
        return False

    print(f"Comparing {len(task_ids)} tasks …\n")

    mismatches: list[int] = []
    identical: list[int] = []

    for task_id in task_ids:
        baseline_file = OUTPUTS_DIR / f"task_{task_id}_baseline_refined.py"
        spec_file = OUTPUTS_DIR / f"task_{task_id}_speculative_refined.py"

        baseline_text = baseline_file.read_text(encoding="utf-8").strip()
        spec_text = spec_file.read_text(encoding="utf-8").strip()

        if baseline_text == spec_text:
            identical.append(task_id)
        else:
            mismatches.append(task_id)
            # Print a short diff hint for the first mismatch
            if len(mismatches) <= 3:
                bl = baseline_text[:120].replace("\n", "↵")
                sp = spec_text[:120].replace("\n", "↵")
                print(f"  task {task_id} MISMATCH")
                print(f"    baseline[0:120]: {bl}")
                print(f"    speculative[0:120]: {sp}\n")

    n_total = len(task_ids)
    n_match = len(identical)
    n_mismatch = len(mismatches)
    mismatch_rate = n_mismatch / n_total if n_total > 0 else 0.0

    print("=" * 60)
    print(f"Results: {n_match}/{n_total} identical, {n_mismatch} mismatches ({mismatch_rate:.1%})")

    verified = n_mismatch == 0

    if verified:
        print("\n[VERIFIED] All outputs identical.")
        print("Speculative decoding is producing correct outputs.")
        print("Safe to publish latency results.")
    else:
        print(f"\n[WARNING] {n_mismatch} mismatch(es) detected.")
        print(f"Affected task IDs: {mismatches}")
        if mismatch_rate > 0.05:
            print("\n[CRITICAL] Mismatch rate > 5%. Do NOT publish until resolved.")
            print("Possible causes:")
            print("  a) temperature is not truly zero in vLLM (check --temperature 0.0 flag)")
            print("  b) MTP implementation has a correctness bug (check PR #41745)")
            print("  c) seed is not applied consistently across streaming chunks")
        else:
            print("\n[MINOR] <5% mismatch rate. Investigate before publishing.")
            print("Note this in your post as a potential floating-point non-determinism issue.")

    # Persist result
    meta = load_metadata()
    meta["output_identity_verified"] = verified
    meta["output_identity_task_count"] = n_total
    meta["output_identity_mismatches"] = n_mismatch
    if mismatches:
        meta["output_identity_mismatch_task_ids"] = mismatches
    save_metadata(meta)
    print(f"\nResult written to {METADATA_PATH}")

    return verified


if __name__ == "__main__":
    ok = verify_output_identity()
    sys.exit(0 if ok else 1)
