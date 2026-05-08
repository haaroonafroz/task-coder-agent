"""
Analysis and visualization for the speculative decoding benchmark (Sections 9–14).

Reads experiments/results.csv and produces:
  Chart 1  – Tokens per second per agent (grouped bar)
  Chart 2  – Pipeline speedup vs output token length (scatter + regression)
  Chart 3  – Latency breakdown per agent (stacked bar)
  Chart 4  – Draft token acceptance rate by agent (bar, speculative only)
  Chart 5  – Pass rate comparison (bar)
  Table    – Aggregate metrics with confidence intervals printed to stdout
             and saved to experiments/aggregate_metrics.json

Usage:
    python analysis/analyze_results.py
    python analysis/analyze_results.py --results path/to/results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS = ROOT / "experiments" / "results.csv"
CHARTS_DIR = ROOT / "experiments" / "charts"
AGGREGATE_OUT = ROOT / "experiments" / "aggregate_metrics.json"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette
C_BASELINE = "#6b7280"    # grey
C_SPECULATIVE = "#2563eb"  # blue
ALPHA_ERR = 0.35


# ---------------------------------------------------------------------------
# Statistics helpers (Section 10.1)
# ---------------------------------------------------------------------------

def compute_stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_95": (float("nan"), float("nan")), "cv": float("nan"), "n": 0}
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=stats.sem(arr)) if n > 1 else (mean, mean)
    cv = std / mean if mean != 0 else float("nan")
    return {"mean": mean, "std": std, "ci_95": (float(ci[0]), float(ci[1])), "cv": cv, "n": n}


def speedup(baseline_mean: float, spec_mean: float) -> float:
    return baseline_mean / spec_mean if spec_mean > 0 else float("nan")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [c for c in df.columns if c not in ("mode", "test_pass", "test_error_message", "acceptance_rate_generator")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["test_pass"] = df["test_pass"].map({"True": True, "False": False, True: True, False: False})
    return df


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_aggregate(df: pd.DataFrame) -> dict:
    agg: dict = {}
    agents = ["planner", "generator", "refiner"]

    for mode in ["baseline", "speculative"]:
        sub = df[df["mode"] == mode]
        for agent in agents:
            col = f"{agent}_decode_ms"
            tok_col = f"{agent}_tokens_generated"
            total_col = f"{agent}_total_ms"
            if col in sub.columns:
                st = compute_stats(sub[col].dropna().tolist())
                agg[f"{agent}_decode_ms_{mode}"] = st
            if tok_col in sub.columns and total_col in sub.columns:
                tps_vals = (sub[tok_col] / (sub[total_col] / 1000.0)).dropna().tolist()
                agg[f"{agent}_tps_{mode}"] = compute_stats(tps_vals)
        pipe_st = compute_stats(sub["pipeline_total_ms"].dropna().tolist())
        agg[f"pipeline_total_ms_{mode}"] = pipe_st
        pass_rate = sub["test_pass"].mean() if "test_pass" in sub.columns else float("nan")
        agg[f"pass_rate_{mode}"] = float(pass_rate)

    for agent in agents:
        bm = agg.get(f"{agent}_decode_ms_baseline", {}).get("mean", float("nan"))
        sm = agg.get(f"{agent}_decode_ms_speculative", {}).get("mean", float("nan"))
        agg[f"{agent}_decode_speedup"] = speedup(bm, sm)

    bm = agg.get("pipeline_total_ms_baseline", {}).get("mean", float("nan"))
    sm = agg.get("pipeline_total_ms_speculative", {}).get("mean", float("nan"))
    agg["pipeline_speedup"] = speedup(bm, sm)

    return agg


def print_aggregate_table(agg: dict) -> None:
    print("\n" + "=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)
    agents = ["planner", "generator", "refiner"]
    for agent in agents:
        bst = agg.get(f"{agent}_decode_ms_baseline", {})
        sst = agg.get(f"{agent}_decode_ms_speculative", {})
        sp = agg.get(f"{agent}_decode_speedup", float("nan"))
        print(f"\n{agent.capitalize()} decode latency:")
        print(f"  Baseline:     {bst.get('mean', float('nan')):.1f} ± {bst.get('std', 0):.1f} ms  "
              f"(95% CI: [{bst.get('ci_95', (0,0))[0]:.1f}, {bst.get('ci_95', (0,0))[1]:.1f}])")
        print(f"  Speculative:  {sst.get('mean', float('nan')):.1f} ± {sst.get('std', 0):.1f} ms  "
              f"(95% CI: [{sst.get('ci_95', (0,0))[0]:.1f}, {sst.get('ci_95', (0,0))[1]:.1f}])")
        print(f"  Speedup:      {sp:.3f}x")

    bst = agg.get("pipeline_total_ms_baseline", {})
    sst = agg.get("pipeline_total_ms_speculative", {})
    sp = agg.get("pipeline_speedup", float("nan"))
    print(f"\nPipeline total:")
    print(f"  Baseline:     {bst.get('mean', float('nan')):.1f} ± {bst.get('std', 0):.1f} ms")
    print(f"  Speculative:  {sst.get('mean', float('nan')):.1f} ± {sst.get('std', 0):.1f} ms")
    print(f"  Speedup:      {sp:.3f}x")
    print(f"\nPass rate:  baseline={agg.get('pass_rate_baseline', float('nan')):.1%}  "
          f"speculative={agg.get('pass_rate_speculative', float('nan')):.1%}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 1: Tokens per second per agent (grouped bar)
# ---------------------------------------------------------------------------

def chart1_tps_per_agent(df: pd.DataFrame) -> None:
    agents = ["planner", "generator", "refiner"]
    labels = ["Planner", "Generator", "Refiner"]

    baseline_means, baseline_stds = [], []
    spec_means, spec_stds = [], []

    for agent in agents:
        tok_col = f"{agent}_tokens_generated"
        total_col = f"{agent}_total_ms"
        for mode, means_list, stds_list in [
            ("baseline", baseline_means, baseline_stds),
            ("speculative", spec_means, spec_stds),
        ]:
            sub = df[df["mode"] == mode]
            tps = (sub[tok_col] / (sub[total_col] / 1000.0)).dropna()
            means_list.append(tps.mean() if len(tps) > 0 else 0.0)
            stds_list.append(tps.std(ddof=1) if len(tps) > 1 else 0.0)

    x = np.arange(len(agents))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_b = ax.bar(x - width / 2, baseline_means, width, yerr=baseline_stds,
                    label="Baseline", color=C_BASELINE, capsize=5, error_kw={"alpha": ALPHA_ERR})
    bars_s = ax.bar(x + width / 2, spec_means, width, yerr=spec_stds,
                    label="Speculative (MTP)", color=C_SPECULATIVE, capsize=5, error_kw={"alpha": ALPHA_ERR})

    for bars in [bars_b, bars_s]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tokens per second")
    ax.set_title("Chart 1 – Tokens/sec per Agent: Baseline vs Speculative Decoding")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, "chart1_tps_per_agent.png")


# ---------------------------------------------------------------------------
# Chart 2: Pipeline speedup vs output token length (scatter + regression)
# ---------------------------------------------------------------------------

def chart2_speedup_vs_tokens(df: pd.DataFrame) -> None:
    token_cols = ["planner_tokens_generated", "generator_tokens_generated", "refiner_tokens_generated"]

    b = df[df["mode"] == "baseline"][["task_id", "pipeline_total_ms"] + token_cols].copy()
    s = df[df["mode"] == "speculative"][["task_id", "pipeline_total_ms"]].copy()
    b.columns = ["task_id", "base_ms"] + token_cols
    s.columns = ["task_id", "spec_ms"]

    merged = b.merge(s, on="task_id")
    merged["total_tokens"] = merged[token_cols].sum(axis=1)
    merged["speedup_ratio"] = merged["base_ms"] / merged["spec_ms"]
    merged = merged.dropna(subset=["total_tokens", "speedup_ratio"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(merged["total_tokens"], merged["speedup_ratio"],
               color=C_SPECULATIVE, alpha=0.65, s=40, label="Individual tasks")

    if len(merged) >= 3:
        x_fit = merged["total_tokens"].values
        y_fit = merged["speedup_ratio"].values
        m, b_coef = np.polyfit(x_fit, y_fit, 1)
        x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
        ax.plot(x_line, m * x_line + b_coef, color="#dc2626", linewidth=2,
                label=f"Linear fit (slope={m:.4f})")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="No speedup (1.0x)")
    ax.set_xlabel("Total tokens generated per task")
    ax.set_ylabel("Speedup ratio (baseline / speculative)")
    ax.set_title("Chart 2 – Pipeline Speedup vs Output Token Length")
    ax.legend()
    fig.tight_layout()
    _save(fig, "chart2_speedup_vs_tokens.png")


# ---------------------------------------------------------------------------
# Chart 3: Latency breakdown per agent (stacked bar)
# ---------------------------------------------------------------------------

def chart3_latency_breakdown(df: pd.DataFrame) -> None:
    modes = ["baseline", "speculative"]
    agents = ["planner", "generator", "refiner"]
    agent_labels = ["Planner decode", "Generator decode", "Refiner decode"]
    colors = ["#93c5fd", "#2563eb", "#1e3a8a"]

    means = {mode: [] for mode in modes}
    for mode in modes:
        sub = df[df["mode"] == mode]
        for agent in agents:
            col = f"{agent}_decode_ms"
            means[mode].append(sub[col].mean() if col in sub.columns else 0.0)

    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(6, 5))
    bottoms = np.zeros(len(modes))

    for idx, (label, color) in enumerate(zip(agent_labels, colors)):
        vals = [means[m][idx] for m in modes]
        ax.bar(x, vals, bottom=bottoms, label=label, color=color)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Speculative (MTP)"])
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Chart 3 – Decode Latency Breakdown by Agent")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "chart3_latency_breakdown.png")


# ---------------------------------------------------------------------------
# Chart 4: Draft token acceptance rate by agent (speculative only)
# ---------------------------------------------------------------------------

def chart4_acceptance_rate(df: pd.DataFrame) -> None:
    spec = df[df["mode"] == "speculative"]
    agents = ["planner", "generator", "refiner"]
    labels = ["Planner", "Generator", "Refiner"]

    rates: list[float] = []
    for agent in agents:
        col = f"acceptance_rate_{agent}"
        if col in spec.columns:
            val = spec[col].dropna()
            rates.append(float(val.mean()) if len(val) > 0 else float("nan"))
        else:
            rates.append(float("nan"))

    # If acceptance rate column is missing (not parsed from vLLM logs),
    # show a placeholder chart with a note.
    has_data = any(not np.isnan(r) for r in rates)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(agents))
    bars = ax.bar(x, rates, color=C_SPECULATIVE, width=0.5)

    for bar, rate in zip(bars, rates):
        if not np.isnan(rate):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{rate:.1%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Draft token acceptance rate")
    ax.set_title("Chart 4 – Draft Token Acceptance Rate by Agent\n(Speculative mode)")
    if not has_data:
        ax.text(0.5, 0.5, "Not available\n(parse from vLLM logs)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="red",
                bbox=dict(boxstyle="round", fc="white", ec="red"))
    fig.tight_layout()
    _save(fig, "chart4_acceptance_rate.png")


# ---------------------------------------------------------------------------
# Chart 5: Pass rate comparison (bar)
# ---------------------------------------------------------------------------

def chart5_pass_rate(df: pd.DataFrame) -> None:
    modes = ["baseline", "speculative"]
    rates = []
    for mode in modes:
        sub = df[df["mode"] == mode]
        rate = sub["test_pass"].mean() if "test_pass" in sub.columns else float("nan")
        rates.append(float(rate))

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = [C_BASELINE, C_SPECULATIVE]
    bars = ax.bar(["Baseline", "Speculative (MTP)"], rates, color=colors, width=0.45)

    for bar, rate in zip(bars, rates):
        if not np.isnan(rate):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{rate:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Pass rate")
    ax.set_title("Chart 5 – Pass Rate: Baseline vs Speculative\n(Must be identical at temperature=0)")
    fig.tight_layout()
    _save(fig, "chart5_pass_rate.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze speculative decoding benchmark results")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS), help="Path to results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)

    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        print("Run both baseline and speculative experiments first.")
        return

    df = load_results(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")
    print(f"Modes present: {df['mode'].unique().tolist()}")

    agg = compute_aggregate(df)
    print_aggregate_table(agg)

    AGGREGATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    AGGREGATE_OUT.write_text(json.dumps(agg, indent=2, default=str))
    print(f"\nAggregate metrics saved to {AGGREGATE_OUT}")

    print("\nGenerating charts …")
    chart1_tps_per_agent(df)
    chart2_speedup_vs_tokens(df)
    chart3_latency_breakdown(df)
    chart4_acceptance_rate(df)
    chart5_pass_rate(df)

    print(f"\nAll charts saved to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
