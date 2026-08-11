// TypeScript types mirroring the Phase 3 FastAPI Pydantic schemas.

export type ModelChoice = "auto" | "local" | "gemini" | "gpt4o";
export type RunKind = "auto" | "new" | "resume" | "repair";

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
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  ts: string;
  run_id: string | null;
  run_kind?: RunKind;
}

export interface Run {
  run_id: string;
  session_id: string;
  request: string;
  status: "queued" | "running" | "completed" | "partial" | "failed" | "error" | "cancelled";
  model: string;
  queued_at: string;
  started_at: string | null;
  finished_at: string | null;
  result: Record<string, unknown> | null;
  error: string | null;
  run_kind: RunKind;
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
