// TypeScript types mirroring the Phase 3 FastAPI Pydantic schemas.

export type ModelChoice = string;
export type RunKind = "auto" | "new" | "resume" | "repair";
export type ExecutionRoute = "auto" | "mission" | "hotfix" | "review";
export type ReviewFixMode = "ask" | "auto";
export type DecisionAction =
  | "apply_fix"
  | "dismiss"
  | "escalate_mission"
  | "run_smoke"
  | "setup_env";

export interface PendingDecision {
  type: string;
  title: string;
  summary: string;
  options: DecisionAction[];
  session_id: string;
  created_at: string;
  payload: Record<string, unknown>;
}
export type WorkspaceKind = "managed" | "external";
export type WorkspaceAccessMode = "read_write" | "read_only";

export interface WorkspaceBinding {
  kind: WorkspaceKind;
  path: string;
  access_mode: WorkspaceAccessMode;
  environment_strategy: "auto" | "project" | "harness";
  git_root: string | null;
  sandbox_required: boolean;
}

export interface ProjectProfile {
  root: string;
  workspace_kind: WorkspaceKind;
  git: Record<string, unknown>;
  languages: string[];
  manifests: string[];
  environment: Record<string, unknown> | null;
  detected_commands: Record<string, string>;
}

export interface WorkspaceInfo {
  workspace: WorkspaceBinding;
  project: ProjectProfile;
}

export interface Session {
  session_id: string;
  title: string;
  status: string;
  selected_model: string;
  thinking_profile: string;
  created_at: string;
  phoenix_session_id: string | null;
  phoenix_project: string | null;
  workspace_root: string;
  plan_path: string;
  events_path: string;
  workspace: WorkspaceBinding;
  project_profile?: Record<string, unknown> | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  ts: string;
  run_id: string | null;
  run_kind?: RunKind;
  execution_route?: ExecutionRoute;
}

export interface Run {
  run_id: string;
  session_id: string;
  request: string;
  status:
    | "queued"
    | "running"
    | "completed"
    | "partial"
    | "failed"
    | "error"
    | "cancelled"
    | "awaiting_decision";
  model: string;
  queued_at: string;
  started_at: string | null;
  finished_at: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
  run_kind: RunKind;
  execution_route?: ExecutionRoute;
  review_fix_mode?: ReviewFixMode;
  plan_id: string | null;
}

export interface Milestone {
  id?: string;
  title?: string;
  description?: string;
  status?: string;
  target_files?: string[];
  validation_contract?: Record<string, unknown>;
}

export interface Plan {
  mission_id?: string;
  title?: string;
  milestones: Milestone[];
}

export interface Handoff {
  milestone_id: string;
  title: string;
  verdict: string;
  worker_summary: string | null;
  files_modified: string[];
  tool_calls: number | null;
  retry_count: number | null;
  commit_hash: string | null;
  timestamp: string | null;
  session_id: string | null;
}

export interface EventData {
  ts: string;
  type: string;
  session_id: string;
  data: Record<string, unknown>;
  index: number;
}

export interface ModelInfo {
  key: string;
  model: string;
  base_url: string;
  available: boolean;
  error: string | null;
  models_by_role?: Record<string, string>;
  thinking_by_role?: Record<string, string>;
  context_length?: number | null;
  label?: string | null;
  adapter?: string | null;
  enabled?: boolean;
  api_key_set?: boolean;
  discovered_models?: string[];
}

export interface ProviderPreset {
  id: string;
  label: string;
  base_url: string;
  adapter: string;
  model?: string;
  models_by_role?: Record<string, string>;
  context_length?: number;
}

export interface MissionsSettingsPayload {
  llm: {
    providers: Array<{
      id: string;
      label: string;
      base_url: string;
      adapter: string;
      model: string;
      models_by_role: Record<string, string>;
      enabled: boolean;
      context_length: number | null;
      api_key: string;
      api_key_set: boolean;
    }>;
    default_provider: string;
    fallback_order: string[];
    seed: number;
    context_length: number;
  };
  roles: Record<string, {
    temperature: number;
    top_p: number;
    max_tokens: number;
    thinking: string;
    thinking_enabled: boolean;
  }>;
  qdrant: {
    mode: "embedded" | "http" | "off";
    path: string;
    url: string;
    api_key: string;
    api_key_set: boolean;
    collection: string;
    dense_name: string;
    sparse_name: string;
    dense_dims: number;
  };
  embeddings: {
    backend: "auto" | "hf" | "openai" | "none";
    hf_model: string;
    openai_model: string;
    openai_dims: number;
  };
  runtime: Record<string, unknown>;
  observability: Record<string, unknown>;
  memory: { backend: string };
}

export interface SettingsResponse {
  settings: MissionsSettingsPayload;
  home: string;
  presets: ProviderPreset[];
  migrated_from_env: boolean;
  container: boolean;
}

export interface ToolParam {
  name: string;
  type: string;
  required: boolean;
  default: unknown;
}

export interface ToolInfo {
  name: string;
  params: ToolParam[];
}

export interface SkillInfo {
  name: string;
  keywords: string[];
}

export interface WorkspaceEntry {
  path: string;
  tree: string;
  entries: string[];
  root?: string;
  nodes?: WorkspaceNode[];
}

export type WorkspaceScope = "workspace" | "session";

export interface WorkspaceNode {
  name: string;
  path: string;
  type: "file" | "directory";
  size?: number | null;
  children?: WorkspaceNode[];
}

export interface WorkspaceFile {
  path: string;
  content: string;
  size: number;
  encoding: string;
}

export interface Upload {
  filename: string;
  size: number;
  path: string;
}

// SSE event shape (one JSON object per data: line)
export interface SSEEvent {
  ts: string;
  type: string;
  session_id: string;
  data: Record<string, unknown>;
  index?: number;
}

export type AgentRole =
  | "orchestrator"
  | "worker"
  | "hotfix"
  | "reviewer"
  | "validator"
  | "verify_hotfix"
  | "triage";

export interface LLMMetrics {
  call_id: string;
  role: AgentRole;
  milestone_id?: string;
  phase?: string;
  model_used: string;
  tokens_prompt: number;
  tokens_generated: number;
  prefill_ms: number;
  decode_ms: number;
  total_ms: number;
  thinking_level?: string;
  output_kind?: string;
  thinking_preview?: string;
  output_preview?: string;
  thinking_chars?: number;
  output_chars?: number;
  fallback_used?: boolean;
}

export interface ToolCallEntry {
  tool: string;
  reasoning?: string;
  ts: string;
  milestone_id?: string;
}

export interface AgentTurn {
  kind: "agent";
  id: string;
  call_id: string;
  role: AgentRole;
  milestone_id?: string;
  phase?: string;
  thinking: string;
  output: string;
  tools: ToolCallEntry[];
  metrics?: LLMMetrics;
  streaming: boolean;
  ts: string;
}

export type ChatItem =
  | { kind: "user"; id: string; ts: string; content: string }
  | { kind: "system"; id: string; ts: string; content: string }
  | {
      kind: "mission_summary";
      id: string;
      ts: string;
      status: string;
      content: string;
      failure_reason?: string;
    }
  | AgentTurn;
