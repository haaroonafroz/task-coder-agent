# Missions Architecture — Self-Improving Multi-Agent Coding Runtime

A serial multi-agent system for building software with **small local LLMs**. The runtime decomposes a user request into milestones, routes a minimal tool set per step, implements code in an isolated workspace, and validates each milestone against an explicit **Validation Contract** before committing. Failed milestones retry with grounded error feedback; structural contract flaws trigger **negotiated replanning** with the Orchestrator.

This repo is a proof of concept: the architecture (not raw model scale) is what makes local 27B-class models usable for non-trivial Python tasks.

---

## Why this architecture exists

Large cloud models tolerate sloppy agent design. Small local models do not. This stack applies standard software-engineering discipline to the agent loop:

| Principle | How it is enforced |
|---|---|
| **Separation of concerns** | Orchestrator plans; Worker implements; Validator adversarially checks. No role shares another role's prompt or tools. |
| **Contract-first delivery** | Every milestone ships with a machine-runnable `validation_contract.command` (pytest, lint, shell). PASS is deterministic before merge. |
| **Negotiation through contracts** | Validator can emit `REPLAN` when the plan itself is wrong (e.g. test command mismatch). Orchestrator patches `plan.json` instead of forcing the Worker to guess. |
| **Anti spec-gaming** | Workers cannot edit test files during implementation milestones. Unauthorized test edits fail immediately with explicit guidance. |
| **Grounded context** | Qdrant injects 2–3 relevant tools per milestone (not all 11). Cognee stores failures and milestone state to reduce repeated hallucinations. |
| **Serial execution** | One LLM call at a time — required on dual 16 GB GPUs where parallel agents would OOM. |
| **Observability** | Arize Phoenix traces Orchestrator / Worker / Validator LLM calls and tool invocations. |

---

## Execution flow

```
User request
     │
     ▼
┌─────────────────────────────────────────┐
│ 1. ORCHESTRATION                        │
│    config/orchestrator.md → plan.json   │
│    Milestones + Validation Contracts    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 2. DYNAMIC SKILL ROUTING (per milestone)│
│    Qdrant hybrid retrieval → top 3 tools│
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 3. WORKER                               │
│    config/worker.md + curated tools     │
│    One tool call at a time → workspace/ │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 4. VALIDATION                           │
│    Run contract command + structural QA │
│    PASS → git commit                    │
│    FAIL → retry Worker (≤3) w/ memory   │
│    REPLAN → Orchestrator patches plan   │
└─────────────────────────────────────────┘
```

On **PASS**, the runtime commits workspace changes and writes handoff JSON to `active_mission/handoffs/`. On crash, `active_mission/plan.json` plus Cognee / JSON memory allow resume.

---

## Project layout

```
config/
  orchestrator.md      # Planning persona + contract rules
  worker.md            # Implementation persona + tool-call format
  validator.md         # Adversarial QA persona
  skills.md            # 11 tools (chunked for Qdrant indexing)

src/
  main.py              # MissionsRuntime — serial pipeline
  llm_client.py        # local → gemini → gpt4o fallback chain
  tool_registry.py     # Cloud Qdrant + BGE embeddings + keyword boost
  memory_layer.py      # Cognee remember/recall + JSON fallback
  telemetry.py         # Arize Phoenix OpenTelemetry
  agents/              # orchestrator, worker, validator phase logic
  tools/               # file_ops, git_ops, system_ops implementations

active_mission/
  plan.json            # Live milestone plan
  handoffs/            # Per-milestone telemetry
  memory_store.json    # Fallback memory when Cognee unavailable

workspace/             # All generated code (sandbox)
test_project.txt       # Copy-paste example mission commands
scripts/
  build_llamacpp.sh    # Build TurboQuant llama.cpp for V100+
  download_models.sh   # HuggingFace GGUF downloads
  start_server_speculative.sh  # MTP llama-server on port 8001
```

Legacy MBPP speculative-decoding benchmark code remains under `pipeline/`, `run_experiment.py`, and `analysis/` for reference.

---

## Requirements

- **OS:** Linux with CUDA (tested on dual Tesla V100 16 GB)
- **Python:** 3.10+
- **GPU driver + CUDA toolkit** (for building llama.cpp with `nvcc`)
- **HuggingFace account** with accepted model licenses (Gemma 4 if used)
- **Optional cloud services:** Qdrant Cloud, OpenAI (embeddings + Cognee), Gemini/GPT-4o fallback

---

## Installation

### 1. Clone and create a virtual environment

```bash
cd coding-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with at minimum:

```ini
HF_TOKEN=hf_...

# Local llama-server (Missions default backend)
LLM_SPECULATIVE_URL=http://localhost:8001/v1
TARGET_MODEL_GGUF=/path/to/models/your-model.gguf
MODEL_ALIAS=your-model-alias
TARGET_MODEL=your-model-alias

# Cognee (uses OpenAI by default for graph extraction)
LLM_API_KEY=sk-...
LLM_PROVIDER=openai

# Qdrant Cloud (skill routing)
QDRANT_URL=https://....cloud.qdrant.io:6333
QDRANT_API_KEY=...
QDRANT_COLLECTION=agent_skills

# Optional fallbacks
GEMINI_API_KEY=...
OPENAI_API_KEY=...
```

### 3. Build llama.cpp (TurboQuant fork)

Pre-built binaries often lack reliable `sm_70` (V100) kernels. Build from source:

```bash
bash scripts/build_llamacpp.sh
export PATH="$HOME/llama-cpp-turboquant/build/bin:$PATH"
```

Add the `export PATH=...` line to your shell profile.

### 4. Download a model

#### Qwen3.6 / Qwopus3.6 (recommended for this repo)

These families ship **MTP draft weights inside a single GGUF file**. No separate draft download is required. The server uses `--spec-type draft-mtp` on that one file.

```bash
# Qwen3.6 27B MTP (default in download script)
bash scripts/download_models.sh --qwen3627b

# Qwopus3.6 27B MTP
bash scripts/download_models.sh --qwopus3627b
```

Copy the printed `TARGET_MODEL_GGUF`, `MODEL_ALIAS`, and `TARGET_MODEL` lines into `.env`.

#### Gemma 4 (separate draft file required)

Gemma 4 MTP uses a **target GGUF plus a separate assistant/draft GGUF**. Download both:

```bash
bash scripts/download_models.sh --e4b   # or --e2b, --26b, --31b
```

Set `TARGET_MODEL_GGUF` and `DRAFT_MODEL_GGUF` in `.env`. For Gemma, uncomment the draft-related lines in `scripts/download_models.sh` and `scripts/start_server_speculative.sh`, and pass the draft model to `llama-server` (e.g. `--model-draft "$DRAFT_FILE"`).

| Family | Draft model | Download |
|---|---|---|
| Qwen3.6 MTP | Built into target GGUF | `--qwen3627b` |
| Qwopus3.6 MTP | Built into target GGUF | `--qwopus3627b` |
| Gemma 4 | Separate assistant GGUF | `--e4b` / `--e2b` / `--26b` / `--31b` |

### 5. Start the local inference server

Only **one** llama-server should run during Missions (serial design + VRAM budget):

```bash
bash scripts/start_server_speculative.sh
curl http://localhost:8001/v1/models   # confirm live
```

The Missions runtime connects to `LLM_SPECULATIVE_URL` (port **8001**) with `model=auto`, which tries local first.

**MTP / speculative decoding** accelerates token generation on supported models, shortening Orchestrator → Worker → Validator iteration cycles on the same hardware.

### 6. Optional: Phoenix observability

```bash
# Terminal 1 — keep running
phoenix serve
# Dashboard: http://127.0.0.1:6006
```

Set in `.env`:

```ini
PHOENIX_HOST=127.0.0.1
PHOENIX_PORT=6006
PHOENIX_EXTERNAL=true
```

Or set `PHOENIX_EXTERNAL=false` to let the runtime launch Phoenix in-process.

---

## Running a mission

### Quick smoke test

```bash
source .venv/bin/activate
python -m src.main "Build a Python module that validates email addresses with tests"
```

### Example projects

`test_project.txt` contains five ready-made missions (arithmetic evaluator, password checker, inventory tracker, text stats, roman numerals). Each line is a full `python -m src.main "..."` command — copy and run.

```bash
# Example: safe math expression evaluator
python -m src.main "Build a Python module that safely evaluates basic math expressions (+, -, *, /, parentheses, integers and floats). Split it into tokenizer.py and evaluator.py. Include tests for valid expressions, division by zero, malformed input, whitespace, and nested parentheses."
```

### CLI options

```bash
python -m src.main --model local   "..."   # force local llama-server
python -m src.main --model gemini  "..."   # force Gemini
python -m src.main --model gpt4o   "..."   # force OpenAI
python -m src.main --model auto    "..."   # local → gemini → gpt4o (default)

python -m src.main --no-telemetry "..."    # disable Phoenix spans
python -m src.main --no-memory    "..."    # disable Cognee / JSON memory
```

### Reset between missions

```bash
rm -f active_mission/plan.json active_mission/memory_store.json
rm -rf active_mission/handoffs/* workspace/*
```

Artifacts after a run:

- **Code:** `workspace/`
- **Plan:** `active_mission/plan.json`
- **Handoffs:** `active_mission/handoffs/M*.json`
- **Server log:** `experiments/logs/speculative_server.log`

---

## Architecture details (local LLM PoC)

### Small context, small tool surface

The Worker never sees all 11 tools. `tool_registry.py` indexes `config/skills.md` into **Qdrant Cloud** with:

- **Dense vectors:** `BAAI/bge-base-en-v1.5` (768-dim, local via HuggingFace)
- **Sparse vectors:** BM25-style TF-IDF with per-skill **keyword boost** from `Keywords:` metadata
- **Fusion:** Reciprocal Rank Fusion → top 3 skills injected per milestone

This reduces tool-selection fatigue — a common failure mode for 7B–27B models given long tool lists.

### Validation contracts and replanning

Each milestone in `plan.json` includes:

```json
"validation_contract": {
  "type": "pytest",
  "command": "python -m pytest tests/test_foo.py -v",
  "pass_criteria": "All tests pass"
}
```

The Validator runs the command first. Structural checks (e.g. illegal test-file edits) short-circuit before an LLM call. If the **plan** is wrong — not the implementation — the Validator returns `REPLAN` and the Orchestrator patches the contract or milestone list.

### Memory and hallucination guardrails

`memory_layer.py` uses **Cognee** (`remember` / `recall`) when installed:

- Milestone completion state for crash recovery
- Validator failure logs as negative constraints on retry
- Structural queries to ground the Worker in prior codebase facts

If Cognee is unavailable, the same data falls back to `active_mission/memory_store.json`.

### Serial execution and VRAM

Parallel multi-agent inference doubles KV-cache pressure and OOMs on 2×16 GB cards. This runtime enforces **one active LLM role at a time**, dedicating the full VRAM budget to whichever agent is running.

---

## Model families and MTP throughput

| Model | MTP draft | Notes |
|---|---|---|
| **Qwen3.6** | In-file (`--spec-type draft-mtp`) | Single GGUF from `unsloth/Qwen3.6-27B-MTP-GGUF` |
| **Qwopus3.6** | In-file | Single GGUF from `Jackrong/Qwopus3.6-27B-v2-MTP-GGUF` |
| **Gemma 4** | Separate assistant GGUF | Target + `*-assistant*` draft; enable draft flags in server script |

Higher throughput from MTP means faster milestone retries and shorter end-to-end missions on the same GPU — important when a 27B local model needs several Validator cycles per feature.

For **Qwen3 thinking models**, the client enables thinking-mode handling in `llm_client.py` so reasoning tokens do not produce empty Worker/Orchestrator output.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` on port 8001 | llama-server not running | `bash scripts/start_server_speculative.sh` |
| Empty Orchestrator output, 80s+ prefill | Qwen thinking tokens exhausted `max_tokens` | Ensure thinking handling is enabled; raise `MAX_TOKENS_ORCHESTRATOR` in `.env` |
| Gemini 400 `Unknown name "seed"` | Gemini rejects OpenAI-only params | Use `--model local` or ensure latest `llm_client.py` strips unsupported params |
| `[Memory] Cognee write failed` | Missing `LLM_API_KEY` for Cognee's internal LLM | Set `LLM_API_KEY` + `LLM_PROVIDER=openai` in `.env` |
| Qdrant connection errors | Bad URL/key or collection missing | Verify `QDRANT_*` vars; router falls back to keyword matching |
| Phoenix dashboard empty | Nothing listening on 6006 | Run `phoenix serve` with `PHOENIX_EXTERNAL=true` |
| Worker edits tests, instant FAIL | Spec-gaming guardrail | Expected — fix implementation, not tests |
| Stale plan resumes wrong mission | Old `plan.json` | Delete `active_mission/plan.json` before a new request |

---

## Legacy: MBPP speculative decoding benchmark

The original research track (`run_experiment.py`, `pipeline/graph.py`, `analysis/analyze_results.py`) compares baseline vs MTP speculative decoding on a 50-task MBPP LangGraph pipeline. It is independent of the Missions runtime:

```bash
bash scripts/start_server_baseline.sh
python run_experiment.py --mode baseline

bash scripts/start_server_speculative.sh
python run_experiment.py --mode speculative

python analysis/analyze_results.py
```

---

## License

See [LICENSE](LICENSE).
