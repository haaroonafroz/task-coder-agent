// Thin fetch wrapper for the Missions Control API.
// All paths are relative ("/api/v1/...") so the Vite dev proxy forwards
// them to the FastAPI server on :8088.

import type {
  Session,
  Message,
  Run,
  Plan,
  Handoff,
  ModelInfo,
  ToolInfo,
  SkillInfo,
  WorkspaceEntry,
  WorkspaceFile,
  Upload,
  ModelChoice,
} from "./types";

const BASE = "/api/v1";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`API ${resp.status}: ${text}`);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json() as Promise<T>;
}

// ---- Sessions ----

export const api = {
  listSessions(status?: string): Promise<Session[]> {
    const q = status ? `?status=${status}` : "";
    return req(`/sessions${q}`);
  },

  createSession(body: { title: string; model?: ModelChoice }): Promise<Session> {
    return req(`/sessions`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  getSession(sid: string): Promise<Session> {
    return req(`/sessions/${sid}`);
  },

  patchSession(sid: string, body: Partial<Session>): Promise<Session> {
    return req(`/sessions/${sid}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    });
  },

  deleteSession(sid: string): Promise<void> {
    return req(`/sessions/${sid}`, { method: "DELETE" });
  },

  // ---- Messages ----

  listMessages(sid: string, limit?: number): Promise<Message[]> {
    const q = limit ? `?limit=${limit}` : "";
    return req(`/sessions/${sid}/messages${q}`);
  },

  createMessage(
    sid: string,
    body: { content: string; trigger_run?: boolean; model?: ModelChoice }
  ): Promise<Message> {
    return req(`/sessions/${sid}/messages`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  // ---- Runs ----

  listRuns(sid: string): Promise<Run[]> {
    return req(`/sessions/${sid}/runs`);
  },

  getRun(sid: string, rid: string): Promise<Run> {
    return req(`/sessions/${sid}/runs/${rid}`);
  },

  // ---- Plan ----

  getPlan(sid: string): Promise<Plan> {
    return req(`/sessions/${sid}/plan`);
  },

  // ---- Handoffs ----

  listHandoffs(sid: string): Promise<Handoff[]> {
    return req(`/sessions/${sid}/handoffs`);
  },

  // ---- Workspace ----

  listWorkspace(sid: string, path?: string, depth?: number): Promise<WorkspaceEntry> {
    const params = new URLSearchParams();
    if (path) params.set("path", path);
    if (depth) params.set("depth", String(depth));
    const q = params.toString() ? `?${params}` : "";
    return req(`/sessions/${sid}/workspace${q}`);
  },

  readWorkspaceFile(sid: string, path: string): Promise<WorkspaceFile> {
    return req(`/sessions/${sid}/workspace/file?path=${encodeURIComponent(path)}`);
  },

  // ---- Models ----

  listModels(): Promise<ModelInfo[]> {
    return req(`/models`);
  },

  // ---- Tools ----

  listTools(): Promise<ToolInfo[]> {
    return req(`/tools`);
  },

  // ---- Skills ----

  listSkills(): Promise<SkillInfo[]> {
    return req(`/skills`);
  },

  // ---- Uploads ----

  listUploads(sid: string): Promise<Upload[]> {
    return req(`/sessions/${sid}/uploads`);
  },

  uploadFile(sid: string, file: File): Promise<Upload> {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/sessions/${sid}/uploads`, {
      method: "POST",
      body: form,
    }).then((r) => {
      if (!r.ok) throw new Error(`Upload failed: ${r.status}`);
      return r.json();
    });
  },

  // ---- Health ----

  health(): Promise<{ status: string }> {
    return req(`/health`);
  },
};
