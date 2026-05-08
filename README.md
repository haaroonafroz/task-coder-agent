# Speculative Decoding Evaluation – Gemma 4 on a Multi-Agent Code Pipeline

Benchmarks the impact of **MTP speculative decoding** (Gemma 4 E4B + E4B-assistant)
on latency, throughput, and correctness when applied to a LangGraph multi-agent
code generation pipeline running on 2 × 16 GB consumer GPUs.

**Organizing claim:**
> Speculative decoding using the Gemma-4 series open-source model running locally delivers
> speedup on a multi-agent coding pipeline running on 2 × 16 GB consumer GPUs,
> with zero change in pass rate — same quality, meaningfully faster.

---

## Project Structure

```
.
├── datasets/
│   ├── prepare_dataset.py       # Generates the frozen 50-task MBPP subset
│   └── mbpp_subset.json         # Written by prepare_dataset.py (not committed)
│
├── pipeline/
│   ├── llm_client.py            # OpenAI-compat client with prefill/decode timing
│   ├── agents.py                # Planner / Generator / Refiner prompt templates
│   ├── graph.py                 # LangGraph pipeline (START→planner→generator→refiner→test→END)
│   └── test_runner.py           # Sandboxed subprocess test executor
│
├── experiments/
│   ├── results.csv              # Per-task metrics for both modes (written at runtime)
│   ├── run_metadata.json        # Reproducibility metadata (written at runtime)
│   ├── outputs/                 # Per-task .txt / .py files (written at runtime)
│   ├── charts/                  # PNG charts (written by analyze_results.py)
│   └── logs/                    # vLLM server stdout logs
│
├── analysis/
│   └── analyze_results.py       # 5 charts + aggregate metrics with CI
│
├── scripts/
│   ├── install_vllm.sh          # One-time environment setup
│   ├── start_server_baseline.sh # vLLM baseline server (port 8000)
│   ├── start_server_speculative.sh  # vLLM MTP server (port 8001)
│   └── stop_server.sh           # Graceful server teardown
│
├── run_experiment.py            # Main benchmark runner
├── verify_outputs.py            # Output identity verification (Section 8)
└── requirements.txt
```

---

## Prerequisites

### Hardware
- 2 × 16 GB VRAM GPUs (32 GB total)
- Linux, no sudo required
- Python virtual environment

### Critical software notes

| Requirement | Why |
|---|---|
| vLLM from PR branch [#41745](https://github.com/vllm-project/vllm/pull/41745) | Gemma 4 MTP support not yet in main release (as of May 2026) |
| `transformers` from git main | `gemma4_assistant` model type requires ≥ 5.8.0 |
| Accept Gemma 4 license on HuggingFace | Required before model download |

---

## Step-by-Step Execution

### Step 0 — One-time setup

```bash
# Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# Run the setup script (installs vLLM, transformers, all deps, logs in to HF)
bash scripts/install_vllm.sh
```

The script will:
1. Clone and install vLLM from the MTP PR branch (or from PyPI if PR merged)
2. Install `transformers` from git main
3. Install all Python dependencies
4. Log you in to HuggingFace
5. Write `datasets/mbpp_subset.json` and print its MD5 hash

**Record the MD5 hash** — it must be identical across both runs.

---

### Step 1 — Calibration run

Before the full benchmark, confirm E4B achieves an acceptable pass rate:

```bash
bash scripts/start_server_baseline.sh

python run_experiment.py --mode baseline --calibration
```

| Pass rate | Action |
|---|---|
| < 30% | Switch to simpler task subset before proceeding |
| 30–60% | Proceed; note model limitations in your post |
| > 60% | Ideal; proceed to full benchmark |

---

### Step 2 — Full baseline run

```bash
# Server should already be running from Step 1
python run_experiment.py \
    --mode baseline \
    --expected-md5 <hash-from-step-0>
```

This runs 3 warmup tasks (discarded) then 50 benchmark tasks through the full
planner → generator → refiner → test pipeline.

Results are appended to `experiments/results.csv`.
Per-task outputs saved to `experiments/outputs/task_<id>_baseline_*.{txt,py}`.

```bash
# Stop baseline server before starting speculative
bash scripts/stop_server.sh baseline
```

---

### Step 3 — Full speculative run

```bash
bash scripts/start_server_speculative.sh
```

**Verify MTP is active** — check the server log for these lines before running:
```
Gemma4 MTP: centroids masking enabled
Gemma4 MTP: draft layer N -> target_layer...
```
If absent, MTP is **not active** and results will be invalid.

```bash
python run_experiment.py \
    --mode speculative \
    --expected-md5 <hash-from-step-0>

bash scripts/stop_server.sh speculative
```

---

### Step 4 — Output identity verification

```bash
python verify_outputs.py
```

At `temperature=0`, speculative decoding **must** produce byte-for-byte identical
outputs to baseline. The script will:
- Compare all 50 `_baseline_refined.py` vs `_speculative_refined.py` files
- Print a per-task diff hint for any mismatch
- Write `output_identity_verified: true/false` to `run_metadata.json`

**Do not publish results if mismatch rate > 5%.**

---

### Step 5 — Analysis and charts

```bash
python analysis/analyze_results.py
```

Produces in `experiments/charts/`:

| File | Content |
|---|---|
| `chart1_tps_per_agent.png` | Tokens/sec per agent, baseline vs speculative (grouped bar) |
| `chart2_speedup_vs_tokens.png` | Pipeline speedup vs output token length (scatter + regression) |
| `chart3_latency_breakdown.png` | Decode latency per agent, stacked (shows where time is spent) |
| `chart4_acceptance_rate.png` | Draft token acceptance rate per agent (speculative only) |
| `chart5_pass_rate.png` | Pass rate comparison (must be identical) |

Aggregate metrics with 95% confidence intervals are printed to stdout and
saved to `experiments/aggregate_metrics.json`.

---

## Experiment Controls (Non-Negotiable)

| Parameter | Value |
|---|---|
| `temperature` | **0** (enables output identity check) |
| `top_p` | 1.0 |
| `seed` | 42 |
| `max_tokens` planner | 250 |
| `max_tokens` generator | 700 |
| `max_tokens` refiner | 700 |
| `num_speculative_tokens` | 4 |
| Dataset | Tasks 11–60 MBPP, same order, MD5-verified |
| Warmup tasks | 3 (discarded before logging) |

---

## Model Pairing

```
TARGET: google/gemma-4-E4B-it
DRAFT:  google/gemma-4-E4B-it-assistant   ← the ONLY correct choice
```

The E4B-assistant is a 4-layer MTP drafter purpose-built for E4B.
Using any other model as the draft (e.g. E2B-it) will fail due to
incompatible hidden state dimensions.

---

## Memory Budget

```
E4B (8-bit):          ~7.5 GB
E4B-assistant (BF16): ~1.2 GB
vLLM KV cache:        ~4.0 GB
Total:               ~12.7 GB   (fits on one card; second card for overflow)
```

`--gpu-memory-utilization 0.85` is set in both server scripts.

---

## Expected Results

| Agent | Decode speedup |
|---|---|
| Planner | 1.1× – 1.3× (short outputs) |
| Generator | **1.5× – 2.0×** (long code, high repetition) |
| Refiner | 1.3× – 1.7× (medium outputs) |
| **Pipeline total** | **1.4× – 1.8×** |

Acceptance rate (Generator): ~60–80%
Pass rate delta: **0%** (must be zero at temperature=0)

---

## Troubleshooting

**No speedup or regression:**
MTP is almost certainly not active. Re-check server logs for the `Gemma4 MTP` lines.
Do not publish results until confirmed.

**Output mismatches at temperature=0:**
- Check that `--temperature 0.0` is actually being applied (not overridden)
- Verify `seed=42` is supported in your vLLM build
- If < 5% mismatch: note as potential floating-point non-determinism

**OOM on GPU:**
- Reduce `--gpu-memory-utilization` to 0.80
- Ensure no other GPU workloads are running
- The baseline server **must** be stopped before starting the speculative server

**`KeyError: 'gemma4_assistant'`:**
Transformers version is too old. Re-run:
```bash
pip install git+https://github.com/huggingface/transformers.git
```

---

## Optional Extensions

- **Iterative repair loop** — add a Repair Agent on test failure (max 2 retries)
- **Quantization comparison** — BF16 vs 8-bit vs 4-bit
- **Context depth sweep** — 512 / 1024 / 2048 / 4096 tokens
- **HumanEval extension** — 30 harder tasks with longer expected outputs
