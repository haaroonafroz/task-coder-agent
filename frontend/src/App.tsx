import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
} from "react";
import type { Message } from "./api/types";
import { api } from "./api/client";
import {
  useSessions,
  useSession,
  useMessages,
  useRuns,
  usePlan,
  useWorkspace,
  useSessionEvents,
} from "./hooks";
import { SessionSidebar } from "./components/SessionSidebar";
import { ChatPanel } from "./components/ChatPanel";
import { RunInspector } from "./components/RunInspector";

const RIGHT_PANEL_MIN = 360;
const RIGHT_PANEL_MAX = 820;
const RIGHT_PANEL_DEFAULT = 520;

export default function App() {
  const [activeSid, setActiveSid] = useState<string | null>(null);
  const [rightPanelWidth, setRightPanelWidth] = useState(() => {
    const saved = window.localStorage.getItem("rightPanelWidth");
    const parsed = saved ? Number(saved) : RIGHT_PANEL_DEFAULT;
    return Number.isFinite(parsed) ? parsed : RIGHT_PANEL_DEFAULT;
  });
  const resizingRef = useRef(false);

  const { sessions, refresh: refreshSessions } = useSessions();
  const { session, setSession } = useSession(activeSid);
  const { messages, sending, sendMessage, appendMessage } = useMessages(activeSid);
  const { runs, refresh: refreshRuns } = useRuns(activeSid);
  const { plan, refresh: refreshPlan } = usePlan(activeSid);
  const { tree, file, fileLoading, refreshTree, openFile } = useWorkspace(activeSid, "session");
  const { events, connected, clearEvents } = useSessionEvents(activeSid);

  // Track files modified in the current run to highlight them.
  const modifiedFilesRef = useRef<Set<string>>(new Set());

  const activeRun = useMemo(
    () => runs.find((r) => r.status === "queued" || r.status === "running") ?? null,
    [runs]
  );

  const handleCancelRun = useCallback(async () => {
    if (!activeSid || !activeRun) return;
    await api.cancelRun(activeSid, activeRun.run_id);
    refreshRuns();
    refreshPlan();
    refreshSessions();
    if (activeSid) {
      api.getSession(activeSid).then(setSession).catch(() => {});
    }
  }, [activeSid, activeRun, refreshRuns, refreshPlan, refreshSessions, setSession]);

  // When events arrive, react to them.
  useEffect(() => {
    if (events.length === 0) return;
    const lastEv = events[events.length - 1];

    switch (lastEv.type) {
      case "session.started":
        refreshSessions();
        break;
      case "plan.created":
      case "plan.updated":
        refreshPlan();
        break;
      case "tool.result": {
        const tool = lastEv.data?.tool as string | undefined;
        if (tool === "write_file" || tool === "patch_file") {
          refreshTree();
        }
        break;
      }
      case "worker.complete":
        refreshTree();
        break;
      case "milestone.passed":
        refreshTree();
        refreshPlan();
        break;
      case "mission.complete":
        refreshTree();
        refreshPlan();
        refreshSessions();
        refreshRuns();
        break;
      case "mission.cancelled":
        refreshTree();
        refreshPlan();
        refreshSessions();
        refreshRuns();
        break;
    }
  }, [events, refreshPlan, refreshTree, refreshSessions, refreshRuns]);

  // Refresh run status periodically when a run is active.
  useEffect(() => {
    if (!activeSid || !activeRun) return;
    const interval = setInterval(() => {
      refreshRuns();
      refreshPlan();
      refreshSessions();
      api.getSession(activeSid).then(setSession).catch(() => {});
      api.listMessages(activeSid).then((msgs) => {
        const latest = msgs[msgs.length - 1];
        if (latest && latest.role === "assistant") {
          appendMessage(latest as Message);
        }
      });
    }, 3000);
    return () => clearInterval(interval);
  }, [activeSid, activeRun, refreshRuns, refreshPlan, refreshSessions, setSession, appendMessage]);

  const handleSend = useCallback(
    async (content: string, triggerRun: boolean, model?: string) => {
      const msg = await sendMessage(content, triggerRun, model);
      if (msg && triggerRun) {
        refreshRuns();
        refreshSessions();
        setTimeout(() => {
          refreshPlan();
          refreshRuns();
        }, 2000);
      }
    },
    [sendMessage, refreshRuns, refreshPlan, refreshSessions]
  );

  const handleSelectSession = useCallback((sid: string) => {
    setActiveSid(sid);
    clearEvents();
    modifiedFilesRef.current.clear();
  }, [clearEvents]);

  const handleResizeStart = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    resizingRef.current = true;
    document.body.classList.add("is-resizing");
  }, []);

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      if (!resizingRef.current) return;
      const nextWidth = Math.min(
        RIGHT_PANEL_MAX,
        Math.max(RIGHT_PANEL_MIN, window.innerWidth - event.clientX)
      );
      setRightPanelWidth(nextWidth);
    };

    const handleMouseUp = () => {
      if (!resizingRef.current) return;
      resizingRef.current = false;
      document.body.classList.remove("is-resizing");
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      document.body.classList.remove("is-resizing");
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem("rightPanelWidth", String(rightPanelWidth));
  }, [rightPanelWidth]);

  const layoutStyle = {
    "--right-panel-width": `${rightPanelWidth}px`,
  } as CSSProperties;

  return (
    <div className="app-layout" style={layoutStyle}>
      {/* Left: Sessions */}
      <SessionSidebar
        sessions={sessions}
        activeSid={activeSid}
        onSelect={handleSelectSession}
        onCreated={refreshSessions}
      />

      {/* Center: Chat */}
      <ChatPanel
        session={session}
        messages={messages}
        sending={sending}
        connected={connected}
        events={events}
        onSend={handleSend}
      />

      <div className="right-resizer" onMouseDown={handleResizeStart} />

      <RunInspector
        plan={plan}
        events={events}
        sessionStatus={session?.status || "idle"}
        activeRun={activeRun}
        tree={tree}
        file={file}
        fileLoading={fileLoading}
        onOpenFile={openFile}
        onRefreshWorkspace={refreshTree}
        onCancelRun={handleCancelRun}
      />
    </div>
  );
}
