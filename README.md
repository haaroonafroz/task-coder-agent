# Missions — Self-Improving Multi-Agent Coding Runtime

A serial multi-agent harness for building and repairing software. A focused
Triage router selects the full Mission pipeline, a scoped Hotfix profile, or a
read-only Code Review profile. Changed code always passes through the Validator;
broad review findings escalate to the Orchestrator.

The harness is **inference-agnostic**. It talks OpenAI-compatible
`/v1/chat/completions` to llama.cpp, Ollama, LM Studio, vLLM, OpenAI, Gemini, or
OpenRouter. It does **not** start a model server. You point it at a local URL or
a cloud API key from the UI or `settings.json`.

This repo is a proof of concept: the architecture (not raw model scale) is what
makes 27B-class local models — and mixed local/cloud setups — usable for
non-trivial coding tasks.

---

## Why this architecture exists

Large cloud models tolerate sloppy agent design. Small local models do not. This
stack applies standard software-engineering discipline to the agent loop:

| Principle | How it is enforced |
|---|---|
| **Separation of concerns** | Triage routes; Orchestrator plans; Worker builds; Hotfix patches locally; Code Review inspects read-only; Validator adversarially checks. Each role has a focused prompt and permissions. |
| **Acceptance-first delivery** | Every milestone ships with observable acceptance criteria and a validation profile. The Validator compiles executable checks; low-level commands never burden the planning model. |
| **Negotiation through packets** | Validator can emit `REPLAN` when packet intent is incomplete or contradictory. Orchestrator emits a small patch op-list (`update_milestone` / `insert` / `remove`) applied deterministically — the whole plan is never regenerated. |
| **Anti spec-gaming** | Harness-owned acceptance tests remain protected, while agent-owned tests may be part of a coherent implementation slice. Workspace writes remain jailed, with `request_scope` available when a packet is incomplete. |
| **Grounded context** | Core inspection/edit/check tools remain available across retries; Qdrant adds niche capabilities. A JSON memory store records failures and milestone state; Cognee is opt-in and fire-and-forget when enabled. |
| **Diff-first editing** | Full `write_file` rewrites of existing files >60 lines are rejected unless the file was read first this milestone (or `rewrite=true`). Keeps decode cost low and regressions rare. |
| **Serial execution** | One LLM call at a time — required on dual 16 GB GPUs where parallel agents would OOM. |
| **Observability** | Arize Phoenix traces Orchestrator / Worker / Validator calls; `llm.call` events with token counts and prefill/decode timing land in `events.jsonl`; the web UI streams session events and session-level token totals in real time. |

---

## Execution flow

```
User request → lifecycle resolution → Triage route
     ├── hotfix → Hotfix → Validator
     ├── review → Code Review → report, Hotfix, or Mission
     └── mission
     │
     ▼
┌─────────────────────────────────────────┐
│ 1. ORCHESTRATION                        │
│    config/orchestrator.md → plan.json   │
│    Milestones + Validation Contracts    │
│    Corrective JSON retries (≤2)        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 1.2 PLAN LINT (deterministic, no LLM)   │
│    Strip workspace/ prefixes, retype    │
│    shell contracts, drop dangling     │
│    depends_on, flag policy-denied cmds │
│    → one Orchestrator patch pass for   │
│    issues the linter can't auto-fix    │
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
│ 3. WORKER (json_mode, batched ≤3 calls) │
│    config/worker.md + curated tools     │
│    Write jail scoped to target_files   │
│    Auto-runs contract after each write │
│    Conversation resumes across retries │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 4. VALIDATION (diff + contract output) │
│    Run contract command + structural QA │
│    Inject bounded git diff to LLM      │
│    PASS → git commit                    │
│    FAIL → retry Worker (≤3) w/ raw     │
│           contract output + memory     │
│    REPLAN → patch ops (budget ≤2,      │
│             fingerprint dedup)         │
└─────────────────────────────────────────┘
```

On **PASS**, the runtime commits workspace changes and writes handoff JSON under
the active session. On crash, the session-local `plan.json`, event log, handoffs,
and memory store allow resume.

---

## Requirements

- **OS:** Linux (macOS works with policy-only sandbox; no bubblewrap)
- **Python:** 3.10+
- **Node.js 18+** (to build the web UI)
- **An OpenAI-compatible LLM endpoint** — llama.cpp, Ollama, LM Studio, vLLM,
  OpenAI, Gemini, or OpenRouter. The harness does **not** start a model server.
- **Optional:** `bubblewrap` (`bwrap`) for a kernel jail on attached folders
- **Optional:** GPU + llama.cpp if you want local inference (see `scripts/`)

---

## Getting started

Inference stays outside this repo. Install the harness, then point it at a `/v1`
URL or a cloud key.

### 1. Clone and install (native, recommended)

Native install can attach any host folder. Docker cannot, unless you bind-mount
it.

```bash
git clone <this-repo> missions
cd missions
bash scripts/install.sh
source .venv/bin/activate
missions doctor
missions serve
```

`install.sh` creates `.venv`, installs the package (`missions` CLI), builds
`frontend/dist`, runs `missions init`, and prints `missions doctor`.

Open **http://127.0.0.1:8088**. On first boot with no providers, a setup wizard
asks for a local OpenAI-compatible URL (for example
`http://127.0.0.1:8001/v1` or Ollama `http://127.0.0.1:11434/v1`) or a cloud API
key.

```bash
missions doctor    # install / config diagnostics
missions init      # write default settings.json (no overwrite unless --force)
```

Debian/Ubuntu jail for attached repos:

```bash
sudo apt install bubblewrap
```

Without bubblewrap, external folders still run, using command policy only.

### 2. Where state lives

Harness state is **not** required to live in the git checkout.

| `TASK_CODER_HOME` | Used when |
|---|---|
| `$TASK_CODER_HOME` | You set it (recommended for a clean install) |
| repository root | The checkout already has `sessions/` or `.env` (developer machine) |
| `~/.missions` | New users with neither of the above |

Under that home directory:

```
settings.json     # non-secret config (providers, roles, qdrant, sandbox)
secrets.json      # API keys (mode 0600) — never commit this
sessions/         # per-session plans, events, handoffs
managed-workspaces/
qdrant/           # embedded vector index (default)
```

Copying `settings.json` / `secrets.json` from another machine is fine. Do **not**
paste `.env` comment suffixes into JSON (for example
`"executor": "auto      # auto | native | bwrap"`). Literals must be a single
token: `"auto"`, `"balanced"`. Restart `missions serve` after hand-editing JSON;
settings are loaded once at process start.

### 3. Manual install (developers)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m playwright install chromium   # optional, ui_smoke
npm --prefix frontend install
npm --prefix frontend run build
missions init
missions serve --reload
```

Existing `.env` files are imported **once** into `settings.json` / `secrets.json`
on first boot. After that, edit JSON or the Settings UI — a `.env` is not
required.

Hot-reload frontend during UI work:

```bash
missions serve --host 127.0.0.1 --port 8088
npm --prefix frontend run dev
```

Vite on `http://127.0.0.1:5173` proxies `/api/*` to port 8088. After changing
the React app, rebuild before expecting `http://127.0.0.1:8088` (which serves
`frontend/dist`) to update:

```bash
npm --prefix frontend run build
```

Then hard-refresh the browser (Ctrl+Shift+R).

### 4. Docker (managed workspaces / cloud-only)

```bash
docker compose up --build
```

Open `http://localhost:8088`. This image does **not** start llama.cpp. Point
Settings at `http://host.docker.internal:11434/v1` (Ollama) or a cloud key.

Docker cannot see arbitrary host paths. To attach a project, bind-mount it and
type the **container** path in the UI:

```yaml
volumes:
  - missions-data:/data
  - ${HOME}/projects:/host-projects
```

Use `/host-projects/my-app` as the workspace path. Prefer the native install if
you work on many local repos.

### 5. Optional local llama.cpp

The TurboQuant / dual-V100 scripts are an **inference sidecar**, not part of
harness install:

```bash
bash scripts/build_llamacpp.sh
bash scripts/download_models.sh --qwen3627b
bash scripts/start_server_speculative.sh
```

Then add or enable provider `local` with `http://127.0.0.1:8001/v1`. Start that
server **before** selecting `local` in the UI. A disabled provider is skipped
(the picker shows it as disabled; Auto uses the enabled fallback chain).

---

## Configure LLMs

A selectable “model” in the UI is a **provider**: `id` + `base_url` + `adapter`
+ `model` name + its own secret. Two OpenAI rows do not share a key.

### Settings UI (preferred)

**Settings → Models**: add a preset (llama.cpp, Ollama, OpenAI, Gemini,
OpenRouter, …), set the model id the server actually serves, paste the API key
on that card, click **Test**. The UI writes `secrets.json` as
`provider.<that-id>` and appends the id to `fallback_order`.

**Settings → Agent**: per-role temperature, max tokens, thinking effort. Those
are *values*. The client maps them onto whatever wire fields the model accepts.

### `settings.json` / `secrets.json`

Minimal OpenAI + local example (keys stay in `secrets.json` only):

```json
{
  "llm": {
    "providers": [
      {
        "id": "local",
        "label": "llama.cpp",
        "base_url": "http://127.0.0.1:8001/v1",
        "adapter": "llamacpp_qwen",
        "model": "qwen3.8-27b-mtp",
        "enabled": true,
        "compat": "auto",
        "context_length": 32768
      },
      {
        "id": "gpt4o",
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "adapter": "openai",
        "model": "gpt-4o",
        "enabled": true,
        "compat": "auto"
      }
    ],
    "default_provider": "auto",
    "fallback_order": ["local", "gpt4o"]
  }
}
```

```json
{
  "provider.gpt4o": "sk-..."
}
```

Rules:

- **`id`** is the dropdown value. **`model`** is the string sent to the API.
- **`models_by_role`** optionally overrides `model` per agent (`worker`,
  `hotfix`, …). Omit it to use one model for every role.
- **`enabled`**: `false` drops the provider from Auto and from runnable
  selections. Re-enable it in Settings when the server is back.
- **`fallback_order`**: Auto walks this list, skipping disabled ids. A new
  provider is **not** used by Auto until you add its id here (the Settings UI
  does that when you add a preset).
- **`adapter`**: `generic` or `openai` (stock Chat Completions),
  `gemini_openai` (Gemini thinking extras), `llamacpp_qwen` (Qwen
  `chat_template_kwargs`).
- **`compat`**: `auto` (default), `classic` (`max_tokens` + sampling), or
  `openai_reasoning` (`max_completion_tokens`, no sampling). `auto` on the
  `openai` adapter treats `gpt-5*` / `gpt-6*` / `o1` / `o3` / `o4` as reasoning
  models. If the API still returns `unsupported_parameter`, the **same**
  provider is retried with that field renamed or dropped.
- Secrets are **`provider.<id>`**, not a shared OpenAI key. `OPENAI_API_KEY` in
  the environment is applied only to ids `gpt4o` and `openai`.
- Custom ids (for example `gpt-6-luna`) are added by editing JSON; the Models
  tab only inserts preset ids. Give that id its own `secrets.json` entry and
  put it on `fallback_order` if Auto should use it.

Hand-edits require a process restart. `missions doctor` lists each provider’s
on/off state and URL.

### Qdrant and embeddings

Default is **embedded** Qdrant via `qdrant-client` 1.18.0 (compatible with
server 1.18.x, including 1.18.2) under `$TASK_CODER_HOME/qdrant`. No Cloud
collection is required. Switch to HTTP/Cloud in Settings → Integrations, or set
`MISSIONS_QDRANT_MODE=http` and `QDRANT_URL`. Embeddings default to local
`BAAI/bge-base-en-v1.5`; keyword ranking is the fallback if the encoder is
unavailable.

---

## Running a mission

### Web UI

```bash
source .venv/bin/activate
missions serve
```

Open `http://127.0.0.1:8088`.

- **Sessions:** create a managed greenfield workspace or attach an existing
  folder (read/write or read-only).
- **Chat:** user messages, streaming agent turns, mission recap. Per-turn stats
  (prefill / generated / context bar) sit under each agent bubble.
- **Run inspector (right):** current stage, **session token totals**, expandable
  **By agent** breakdown (tokens + tool counts), milestone events, workspace
  files.
- **Stop:** cancels the current run cooperatively at the next agent checkpoint
  (does not kill `missions serve` or the model server).
- **Settings:** providers, role knobs, Qdrant, sandbox.

Mockups of the control UI (not live captures):

![Missions Control UI overview](docs/screenshots/control-ui-overview.svg)

![Run inspector milestone and workspace detail](docs/screenshots/run-inspector-detail.svg)

Theme is a dark control UI. Sensitive files (`.env`, keys) are hidden in the
workspace browser.

### CLI

```bash
python -m src.main "Build a Python module that validates email addresses with tests"

python -m src.main --model local   "..."   # one provider id
python -m src.main --model gpt4o   "..."
python -m src.main --model auto    "..."   # walks fallback_order

python -m src.main --workspace /home/me/projects/billing-api \
  "Fix the failing invoice test"

python -m src.main --workspace /home/me/projects/billing-api --read-only \
  --execution-route review "Review this project for correctness bugs"
```

`test_project.txt` has copy-paste example missions.

### Repair / resume

A follow-up on a completed session starts a `repair` run in the same workspace.
Plans are archived under `sessions/<id>/plans/`.

```bash
python -m src.main --session <session-id> --run-kind repair \
  "Fix the runtime error and add a regression test"
python -m src.main --session <session-id> --run-kind resume \
  "Continue the interrupted run"
```

Deleting a session never deletes an attached external workspace. Managed
workspaces are deleted with their session.

```bash
rm -rf "$TASK_CODER_HOME/sessions/<session-id>" \
       "$TASK_CODER_HOME/managed-workspaces/<session-id>"
```

Artifacts after a run:

- **Managed code:** `managed-workspaces/<session-id>/`
- **External code:** remains at the attached path
- **Plan / events / handoffs / runs:** under `sessions/<session-id>/`

---

## Project layout

```
config/                # Agent personas and skills.md (Qdrant index)
src/
  cli.py               # missions serve | init | doctor
  settings/            # schema + settings.json / secrets.json store
  llm_client.py        # named providers, adapters, request-shape compat
  tool_registry.py     # embedded or HTTP Qdrant + BGE / keyword fallback
  api/                 # FastAPI: sessions, runs, events, settings, files
  agents/              # triage, orchestrator, worker, hotfix, review, validator
  sandbox/             # bubblewrap or native+policy
frontend/              # React control UI (build → frontend/dist)
scripts/install.sh     # Native install
Dockerfile             # App-only image (no inference)
```

Session state lives under `$TASK_CODER_HOME` (see above), not necessarily in
this tree.

Legacy MBPP speculative-decoding benchmark code remains under `pipeline/`,
`run_experiment.py`, and `analysis/` for reference.

---

## Architecture details

### Small context, small tool surface

The Worker receives a small persistent core surface and can discover niche
capabilities. `tool_registry.py` indexes `config/skills.md` into Qdrant
(embedded by default, or HTTP/Cloud) with:

- **Dense vectors:** `BAAI/bge-base-en-v1.5` (768-dim, local via HuggingFace)
- **Sparse vectors:** BM25-style TF-IDF with per-skill **keyword boost** from `Keywords:` metadata
- **Fusion:** Reciprocal Rank Fusion → top 3 niche skills injected per milestone

This reduces tool-selection fatigue — a common failure mode for 7B–27B models given long tool lists.

### Worker protocol (json_mode + batched calls)

The Worker requests structured JSON mode and applies deterministic parsing and
repair when providers return malformed output. Each turn the worker emits one
JSON object:

- A single tool call: `{"tool": "...", "args": {...}, "reasoning": "..."}`
- A batch of up to 3 calls: `{"calls": [...]}` — one LLM round trip instead of three
- A status signal: `{"status": "complete" | "blocked" | "request_scope"}`

Native chat messages (system + alternating user/assistant) are sent to the LLM —
never a flattened single-turn blob — so the rendered prompt stays append-only and
llama.cpp's prefix cache stays hot across turns. On validator FAIL, the
conversation **resumes** (not cold-restarts) with the validator's raw contract
output appended.

### Progressive tool discovery and enforcement

The runtime always exposes file inspection/editing, project discovery and
verification tools. Qdrant adds specialized capabilities plus the
always-available deterministic `search_tools` meta-tool:

```json
{
  "tool": "search_tools",
  "args": {"query": "run a targeted Python import smoke test", "limit": 3},
  "reasoning": "The current tool set does not cover this operation."
}
```

The router performs dense+sparse retrieval without a second LLM. Newly
discovered tools are appended to the conversation and become callable on the
next Worker turn. The active tool set is enforced at runtime, and every tool
call is checked against a deterministic argument schema. Unknown tools and
wrong argument names receive actionable correction messages instead of being
executed.

Tool failures are classified and emitted to `events.jsonl` with bounded,
redacted stdout/stderr tails, return codes, timeout/policy status, duration,
and a stable failure signature. Repeating the same failure twice or making
five consecutive failed calls trips the Worker circuit breaker.

### Harness-side validation feedback

After every successful `write_file` / `patch_file`, the harness automatically
runs the Validator-compiled test or lint check and appends the result to the
worker's next turn. UI checks are run by the Validator after the worker
signals completion.

### Write jail + diff-first enforcement

- **Per-milestone write policy**: `write_file` and `patch_file` reject paths outside the packet's declared scope at the tool layer; the worker can return `request_scope` when a legitimate helper or manifest is missing.
- **Diff-first**: full `write_file` rewrites of existing files >60 lines are rejected unless the file was read first this milestone, or `"rewrite": true` is passed.

### Acceptance criteria and replanning

Each milestone in `plan.json` includes high-level intent:

```json
"acceptance_criteria": ["The feature behaves correctly."],
"validation_profile": "python"
```

The Validator compiles the profile and criteria into canonical checks. Structural checks (e.g. illegal test-file edits, out-of-scope writes) short-circuit before an LLM call. The LLM validator receives the **bounded git diff** of actual changes, not just the worker's self-report. Every verdict carries a **failure signature** for replan dedup.

If the **plan** is wrong — not the implementation — the Validator returns `REPLAN` and the Orchestrator emits a **patch op-list**:

```json
{"operations": [
  {"op": "update_milestone", "milestone_id": "M3", "fields": {"acceptance_criteria": ["..."], "validation_profile": "ui"}},
  {"op": "insert_milestone_after", "after_id": "M2", "milestone": {...}},
  {"op": "remove_milestone", "milestone_id": "M4"}
]}
```

`plan_ops.py` validates and applies these deterministically — completed milestones are immutable, ids stay unique, `depends_on` references are cleaned up. The whole plan is never regenerated.

### Plan lint (pre-execution)

Before milestone 1 runs, `plan_lint.py` deterministically:

- Strips `workspace/` prefixes from paths and contract commands
- Retypes `shell` contracts matching `pytest`/`flake8`/`py_compile` patterns to their typed forms (avoids exit-127 loops from shell execution)
- Drops invalid `depends_on` references
- Flags environment-setup milestones, policy-denied commands, and missing contract fields

Issues the linter can't auto-fix go to the Orchestrator for one patch-ops repair pass.

### Replan circuit breakers

- **Replan budget**: `MAX_REPLANS_PER_MILESTONE` (default 2) consecutive replans per milestone, then halt.
- **Fingerprint dedup**: if the same failure signature recurs, the runtime halts immediately — an identical failure with a paraphrased replan is always futile.
- **Re-anchoring by milestone ID**: after a patch-based replan (which can insert/remove milestones), the loop finds the current milestone by ID, not index arithmetic.

### Memory and hallucination guardrails

`memory_layer.py` uses a **JSON file store** as the synchronous source of truth (`sessions/<id>/memory_store.json`):

- Milestone completion state for crash recovery
- Validator failure logs as negative constraints on retry
- Structural queries to ground the Worker in prior codebase facts

**Cognee** is opt-in (`MISSIONS_MEMORY_BACKEND=cognee` or Settings → memory backend). When enabled, writes are **fire-and-forget** on a background event loop — they never block the serial pipeline. The JSON store is always written first for durability.

### Serial execution and VRAM

Parallel multi-agent inference doubles KV-cache pressure and OOMs on 2×16 GB cards. This runtime enforces **one active LLM role at a time**, dedicating the full VRAM budget to whichever agent is running.

### Model families, MTP, and thinking

| Model | MTP draft | Notes |
|---|---|---|
| **Qwen3.6** | In-file (`--spec-type draft-mtp`) | Single GGUF from `unsloth/Qwen3.6-27B-MTP-GGUF` |
| **Qwopus3.6** | In-file (`--spec-type draft-mtp`)| Single GGUF from `Jackrong/Qwopus3.6-27B-v2-MTP-GGUF` |
| **Gemma 4** | Separate assistant GGUF | Target + `*-assistant*` draft; enable draft flags in server script |

Higher throughput from MTP means faster milestone retries and shorter end-to-end missions on the same GPU.

For **Qwen3 thinking models**, adapter `llamacpp_qwen` sends
`chat_template_kwargs.enable_thinking` from Settings → Agent (per-role
`thinking` / `thinking_enabled`). The `/no_think` prompt prefix is silently
ignored by Qwen jinja templates; without the kwarg the model can emit hundreds
of hidden reasoning tokens per turn. Gemini uses `thinking_config` on adapter
`gemini_openai`. Newer OpenAI reasoning models use `compat: auto` (or
`openai_reasoning`) as described under Configure LLMs.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `missions doctor` ValidationError on `sandbox.executor` / `mode` | Copied `settings.json` still has `.env` comments in the string | Use `"auto"` / `"balanced"` only; upgrade past the load sanitizer, or edit the two fields |
| Provider in the dropdown shows **(unavailable)** | `GET {base_url}/models` failed (usually missing key) | Put the key in `secrets.json` as `provider.<id>`; Test in Settings |
| Provider shows **(disabled)** / `Unknown or disabled provider` | `"enabled": false` or session pinned to that id | Enable it in Settings, or pick Auto / another provider |
| `Unsupported parameter: max_tokens` | gpt-5 / gpt-6 / o-series Chat Completions | Leave `compat` on `auto`; restart serve so the client remap is loaded |
| `Connection refused` on port 8001 | llama-server not running | `bash scripts/start_server_speculative.sh` (or your own `/v1` server) |
| Empty Worker output, slow turns (Qwen) | Thinking not disabled for the worker | Settings → Agent → worker thinking off; adapter `llamacpp_qwen` |
| Gemini 400 `Unknown name "seed"` | Gemini rejects OpenAI-only params | Adapter `gemini_openai` strips `seed` / `stream_options` |
| Session token totals missing in the inspector | UI is the old `frontend/dist` bundle | `npm --prefix frontend run build`, hard-refresh 8088 |
| `[Memory] Cognee write scheduling failed` | Missing key for Cognee's internal LLM | Set Cognee LLM env only when memory backend is `cognee` |
| Qdrant connection errors | HTTP/Cloud URL or key wrong | Settings → Integrations; embedded mode needs no URL. Keyword fallback still works |
| Phoenix dashboard empty | Nothing listening on 6006 | `phoenix serve` or disable API telemetry in Settings |
| Worker edits tests, instant FAIL | Spec-gaming guardrail | Expected — fix implementation, not tests |
| `REWRITE REJECTED` on write_file | Diff-first enforcement | `read_file` the target first, then `patch_file`; or pass `"rewrite": true` |
| `MILESTONE BOUNDARY BREACH` | Write jail | Only write to files listed in the milestone's `target_files` |
| `replan_budget_exhausted` | Replan circuit breaker | Same `failure_signature` recurred; fix the plan |
| Stale plan resumes wrong mission | Old session state | New session in the UI, or delete `sessions/<id>/` |
| Hand-edited `settings.json` ignored | Process already loaded settings | Restart `missions serve` |

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
