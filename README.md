# Speculative Decoding Benchmark — Gemma 4 Multi-Agent Code Pipeline

Tests MTP speculative decoding (Gemma 4 target + assistant drafter) against baseline autoregressive decoding on a LangGraph code generation pipeline. Measures latency, throughput, and pass rate across 50 MBPP tasks.

---

## Project Structure

```
datasets/prepare_dataset.py          # writes mbpp_subset.json + MD5
pipeline/llm_client.py               # streaming client, splits prefill/decode timing
pipeline/agents.py                   # prompt templates (Planner / Generator / Refiner)
pipeline/graph.py                    # LangGraph DAG
pipeline/test_runner.py              # sandboxed subprocess executor
run_experiment.py                    # main runner (calibration + full benchmark)
verify_outputs.py                    # byte-for-byte output identity check
analysis/analyze_results.py          # 5 charts + aggregate metrics
scripts/start_server_baseline.sh     # vLLM port 8000
scripts/start_server_speculative.sh  # vLLM port 8001 with --speculative-config
scripts/stop_server.sh               # teardown + GPU memory release
```

---

## Requirements

**RunPod setup:**
- As of date of Experiments (May, 2026)
- GPU with CUDA 12.9+ driver (RTX 5090 guarantees this; older cards may not)
- Container image: `runpod/pytorch:2.11.0-py3.11-cuda12.9-devel-ubuntu22.04`

> vLLM 0.20.1 only publishes `cu129`/`cu13` wheels. Pods with CUDA < 12.9 will fail with "driver too old".

**Install:**
```bash
# PyTorch first — vLLM links against it at install time
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu129

pip install -r requirements.txt
```

**Verify:**
```bash
python -c "import torch, vllm; print(torch.cuda.is_available(), torch.version.cuda, vllm.__version__)"
# Expected: True 12.9 0.20.1
```

**HuggingFace token** (Gemma 4 models are gated):
```bash
cp .env.example .env   # set HF_TOKEN=hf_...
# Accept license at: https://huggingface.co/google/gemma-4-E4B-it
```

---

## Gemma 4 Model Reference

All four variants have Google-published MTP assistant (drafter) models. Every size supports the full speculative decoding benchmark.

### Target + assistant model IDs

| Model | Target | Assistant | Assistant size |
|---|---|---|---|
| E2B | `google/gemma-4-E2B-it` | `google/gemma-4-E2B-it-assistant` | ~78M params / 157 MB |
| **E4B** *(default)* | `google/gemma-4-E4B-it` | `google/gemma-4-E4B-it-assistant` | ~78M params / 159 MB |
| 26B A4B | `google/gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it-assistant` | ~420M params / 839 MB |
| 31B | `google/gemma-4-31B-it` | `google/gemma-4-31B-it-assistant` | ~470M params / 939 MB |

### VRAM requirements (weights + assistant + KV cache)

| Model | BF16 | INT8 | INT4 | Assistant (BF16) | Min GPU |
|---|---|---|---|---|---|
| E2B | ~10 GB | ~5 GB | ~4 GB | +0.3 GB | 8 GB |
| **E4B** | ~16 GB | **~8 GB** | ~5 GB | +0.3 GB | **24 GB** |
| 26B A4B | ~54 GB | ~31 GB | ~22 GB | +1 GB | 32 GB (INT4→24 GB) |
| 31B | ~65 GB | ~37 GB | ~24 GB | +1 GB | 40 GB (INT4→24 GB) |

> KV cache adds ~4 GB at 4096 context. 26B A4B at INT4 fits a 24 GB card and delivers near-31B quality — best option if upgrading from E4B.

### Quantization flags for vLLM

```bash
# BF16 (no quantization)
--dtype bfloat16

# INT8 — default in this repo
--dtype bfloat16 --quantization bitsandbytes

# INT4 — use when INT8 OOMs
--dtype bfloat16 --quantization bitsandbytes --load-in-4bit
```

---

## Running the Benchmark

### Step 0 — Prepare dataset

```bash
python datasets/prepare_dataset.py
# Prints MD5 hash — save it, pass it to both runs below
```

### Step 1 — Baseline (port 8000)

```bash
bash scripts/start_server_baseline.sh
curl http://localhost:8000/v1/models          # confirm live

# Optional calibration first — checks model pass rate before committing
python run_experiment.py --mode baseline --calibration

# Full run
python run_experiment.py --mode baseline --expected-md5 <md5>
bash scripts/stop_server.sh baseline          # stop before starting speculative
```

Calibration pass rate guide:

| Pass rate | Action |
|---|---|
| > 60% | Proceed |
| 30–60% | Proceed, note capability floor |
| < 30% | Switch to a larger model or simpler tasks (`--tasks 20`) |

### Step 2 — Speculative (port 8001)

```bash
bash scripts/start_server_speculative.sh

# Confirm MTP is active — do not skip this
grep "Gemma4 MTP" experiments/logs/speculative_server.log
# Must show: "Gemma4 MTP: centroids masking enabled"

python run_experiment.py --mode speculative --expected-md5 <md5>
bash scripts/stop_server.sh speculative
```

### Step 3 — Verify + Analyze

```bash
python verify_outputs.py          # all outputs must be byte-identical at temperature=0
python analysis/analyze_results.py
```

Charts saved to `experiments/charts/`. Metrics with 95% CI saved to `experiments/aggregate_metrics.json`.

---

## Quick Reference

```bash
cp .env.example .env && nano .env                    # set HF_TOKEN
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
python datasets/prepare_dataset.py                   # save MD5

bash scripts/start_server_baseline.sh
python run_experiment.py --mode baseline --calibration
python run_experiment.py --mode baseline --expected-md5 <md5>
bash scripts/stop_server.sh baseline

bash scripts/start_server_speculative.sh
grep "Gemma4 MTP" experiments/logs/speculative_server.log
python run_experiment.py --mode speculative --expected-md5 <md5>
bash scripts/stop_server.sh speculative

python verify_outputs.py
python analysis/analyze_results.py
```

---

## Switching Models

Edit `MODEL` and `DRAFT_MODEL` at the top of both server scripts:

```bash
# scripts/start_server_baseline.sh and start_server_speculative.sh

# E2B
MODEL="google/gemma-4-E2B-it"
DRAFT_MODEL="google/gemma-4-E2B-it-assistant"

# E4B (default)
MODEL="google/gemma-4-E4B-it"
DRAFT_MODEL="google/gemma-4-E4B-it-assistant"

# 26B A4B — use INT4 on 24 GB, INT8 on 32 GB
MODEL="google/gemma-4-26B-A4B-it"
DRAFT_MODEL="google/gemma-4-26B-A4B-it-assistant"

# 31B — use INT4 on 24 GB, INT8 on 40 GB+
MODEL="google/gemma-4-31B-it"
DRAFT_MODEL="google/gemma-4-31B-it-assistant"
```

For 26B A4B and 31B on tight VRAM, also reduce these flags in the scripts:
```bash
--gpu-memory-utilization 0.90   # was 0.85
--max-model-len 2048            # was 4096
```

---

## Experiment Controls

| Parameter | Value |
|---|---|
| `temperature` | 0 |
| `top_p` | 1.0 |
| `seed` | 42 |
| `max_tokens` planner / generator / refiner | 250 / 700 / 700 |
| `num_speculative_tokens` | 4 |
| Dataset | 50 MBPP tasks, IDs 11–60, fixed order, MD5-verified |
| Warmup tasks | 3 (discarded) |
| Quantization default | BitsAndBytes INT8 |

---

## Expected Results

| Agent | Decode speedup |
|---|---|
| Planner | 1.1× – 1.3× |
| Generator | 1.5× – 2.0× |
| Refiner | 1.3× – 1.7× |
| **Pipeline total** | **1.4× – 1.8×** |

Pass rate delta must be **0%** at temperature=0.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `NVIDIA driver too old` | Pod host driver < 12.9. Use RTX 5090 on RunPod (guarantees new driver) or install vLLM `cu124` wheel for older pods |
| `does not support multimodal models` | vLLM ≤ 0.19.0. Upgrade to 0.20.1 |
| `KeyError: 'gemma4_assistant'` | `pip install git+https://github.com/huggingface/transformers.git` |
| No speedup / speedup ≈ 1.0× | MTP not active. Check log for `Gemma4 MTP: centroids masking enabled` |
| OOM | Lower `--gpu-memory-utilization` to 0.80. Stop baseline server before starting speculative |
| Output mismatches > 5% | Do not publish. Check `temperature=0` and `seed=42` are applied |
