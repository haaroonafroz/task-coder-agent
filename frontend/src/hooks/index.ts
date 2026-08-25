// React hooks for the Missions Control UI.

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { openEventStream } from "../api/events";
import type {
  Session,
  Message,
  Run,
  Plan,
  WorkspaceEntry,
  WorkspaceFile,
  SSEEvent,
  WorkspaceScope,
} from "../api/types";

// ---- Sessions ----

export function useSessions() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const list = await api.listSessions();
      setSessions(list);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { sessions, loading, error, refresh };
}

// ---- Session detail ----

export function useSession(sid: string | null) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!sid) {
      setSession(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    api.getSession(sid).then((s) => {
      if (!cancelled) {
        setSession(s);
        setLoading(false);
      }
    }).catch(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [sid]);

  return { session, loading, setSession };
}

// ---- Messages ----

export function useMessages(sid: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sending, setSending] = useState(false);

  const refresh = useCallback(async () => {
    if (!sid) return;
    try {
      const list = await api.listMessages(sid);
      setMessages(list);
    } catch {
      // ignore
    }
  }, [sid]);

  useEffect(() => {
    setMessages([]);
    if (sid) refresh();
  }, [sid, refresh]);

  const sendMessage = useCallback(
    async (content: string, triggerRun: boolean, model?: string) => {
      if (!sid) return null;
      setSending(true);
      try {
        const msg = await api.createMessage(sid, {
          content,
          trigger_run: triggerRun,
          model: model as "auto" | "local" | "gemini" | "gpt4o" | undefined,
        });
        setMessages((prev) => [...prev, msg]);
        return msg;
      } finally {
        setSending(false);
      }
    },
    [sid]
  );

  const appendMessage = useCallback((msg: Message) => {
    setMessages((prev) => {
      if (prev.some((m) => m.id === msg.id)) return prev;
      return [...prev, msg];
    });
  }, []);

  return { messages, sending, sendMessage, refresh, appendMessage };
}

// ---- Runs ----

export function useRuns(sid: string | null) {
  const [runs, setRuns] = useState<Run[]>([]);

  const refresh = useCallback(async () => {
    if (!sid) return;
    try {
      const list = await api.listRuns(sid);
      setRuns(list);
    } catch {
      // ignore
    }
  }, [sid]);

  useEffect(() => {
    setRuns([]);
    if (sid) refresh();
  }, [sid, refresh]);

  return { runs, refresh };
}

// ---- Plan ----

export function usePlan(sid: string | null) {
  const [plan, setPlan] = useState<Plan | null>(null);

  const refresh = useCallback(async () => {
    if (!sid) return;
    try {
      const p = await api.getPlan(sid);
      setPlan(p);
    } catch {
      // ignore
    }
  }, [sid]);

  useEffect(() => {
    setPlan(null);
    if (sid) refresh();
  }, [sid, refresh]);

  return { plan, refresh };
}

// ---- Workspace ----

export function useWorkspace(sid: string | null, scope: WorkspaceScope = "workspace") {
  const [tree, setTree] = useState<WorkspaceEntry | null>(null);
  const [file, setFile] = useState<WorkspaceFile | null>(null);
  const [fileLoading, setFileLoading] = useState(false);

  const refreshTree = useCallback(async () => {
    if (!sid) return;
    try {
      const t = await api.listWorkspace(sid, "", 4, scope);
      setTree(t);
    } catch {
      // ignore
    }
  }, [sid, scope]);

  const openFile = useCallback(
    async (path: string) => {
      if (!sid) return;
      setFileLoading(true);
      try {
        const f = await api.readWorkspaceFile(sid, path, scope);
        setFile(f);
      } catch {
        setFile(null);
      } finally {
        setFileLoading(false);
      }
    },
    [sid, scope]
  );

  useEffect(() => {
    setTree(null);
    setFile(null);
    if (sid) refreshTree();
  }, [sid, refreshTree]);

  return { tree, file, fileLoading, refreshTree, openFile };
}

// ---- SSE Events ----

export function useSessionEvents(sid: string | null) {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    // Close previous stream
    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }
    setEvents([]);

    if (!sid) {
      setConnected(false);
      return;
    }

    const cleanup = openEventStream(
      sid,
      (event) => {
        setEvents((prev) => [...prev, event]);
      },
      () => {
        setConnected(false);
      }
    );
    cleanupRef.current = cleanup;
    setConnected(true);

    return () => {
      cleanup();
      cleanupRef.current = null;
    };
  }, [sid]);

  const clearEvents = useCallback(() => setEvents([]), []);

  return { events, connected, clearEvents };
}

// ---- Models catalog ----

export function useModels() {
  const [models, setModels] = useState<import("../api/types").ModelInfo[]>([]);

  useEffect(() => {
    api.listModels().then(setModels).catch(() => setModels([]));
  }, []);

  return { models };
}
