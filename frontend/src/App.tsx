import { useCallback, useEffect, useRef, useState } from "react";
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
import { ProgressPanel } from "./components/ProgressPanel";
import { WorkspaceExplorer } from "./components/WorkspaceExplorer";
import { EventDrawer } from "./components/EventDrawer";

type RightTab = "progress" | "workspace" | "events";

export default function App() {
  const [activeSid, setActiveSid] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<RightTab>("progress");

  const { sessions, loading: sessionsLoading, refresh: refreshSessions } = useSessions();
  const { session, setSession } = useSession(activeSid);
  const { messages, sending, sendMessage, appendMessage } = useMessages(activeSid);
  const { runs, refresh: refreshRuns } = useRuns(activeSid);
  const { plan, refresh: refreshPlan } = usePlan(activeSid);
  const { tree, file, fileLoading, refreshTree, openFile } = useWorkspace(activeSid);
  const { events, connected, clearEvents } = useSessionEvents(activeSid);

  // Track files modified in the current run to highlight them.
  const modifiedFilesRef = useRef<Set<string>>(new Set());

  // When events arrive, react to them.
  useEffect(() => {
    if (events.length === 0) return;
    const lastEv = events[events.length - 1];

    switch (lastEv.type) {
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
        // Append assistant summary from the run result
        if (lastEv.data) {
          const status = lastEv.data.status as string;
          const passed = lastEv.data.milestones_passed as number;
          const total = lastEv.data.milestones_total as number;
          const summary = `Mission complete — status: ${status}, milestones ${passed}/${total} passed.`;
          // The run queue already appends a message, but we also add it locally
          // in case the SSE arrives before the message store write.
        }
        break;
    }
  }, [events, refreshPlan, refreshTree, refreshSessions, refreshRuns]);

  // Refresh run status periodically when a run is active.
  useEffect(() => {
    if (!activeSid || runs.length === 0) return;
    const latestRun = runs[0];
    if (latestRun.status === "queued" || latestRun.status === "running") {
      const interval = setInterval(() => {
        refreshRuns();
        api.listMessages(activeSid).then((msgs) => {
          // Check for new assistant messages
          const latest = msgs[msgs.length - 1];
          if (latest && latest.role === "assistant") {
            appendMessage(latest as Message);
          }
        });
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [activeSid, runs, refreshRuns, appendMessage]);

  const handleSend = useCallback(
    async (content: string, triggerRun: boolean, model?: string) => {
      const msg = await sendMessage(content, triggerRun, model);
      if (msg && triggerRun) {
        refreshRuns();
      }
    },
    [sendMessage, refreshRuns]
  );

  const handleSelectSession = useCallback((sid: string) => {
    setActiveSid(sid);
    clearEvents();
    modifiedFilesRef.current.clear();
  }, [clearEvents]);

  return (
    <div className="app-layout">
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
        onSend={handleSend}
      />

      {/* Right: Tabbed panel (Progress / Workspace / Events) */}
      <div className="panel" style={{ overflow: "hidden" }}>
        <div className="tabs">
          <div
            className={`tab ${rightTab === "progress" ? "active" : ""}`}
            onClick={() => setRightTab("progress")}
          >
            Progress
          </div>
          <div
            className={`tab ${rightTab === "workspace" ? "active" : ""}`}
            onClick={() => setRightTab("workspace")}
          >
            Workspace
          </div>
          <div
            className={`tab ${rightTab === "events" ? "active" : ""}`}
            onClick={() => setRightTab("events")}
          >
            Events
          </div>
        </div>

        <div style={{ flex: 1, overflow: "hidden" }}>
          {rightTab === "progress" && (
            <ProgressPanel
              plan={plan}
              events={events}
              sessionStatus={session?.status || "idle"}
            />
          )}
          {rightTab === "workspace" && (
            <WorkspaceExplorer
              tree={tree}
              file={file}
              fileLoading={fileLoading}
              onOpenFile={openFile}
              onRefresh={refreshTree}
            />
          )}
          {rightTab === "events" && <EventDrawer events={events} />}
        </div>
      </div>
    </div>
  );
}
