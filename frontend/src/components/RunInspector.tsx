import { useMemo, useState } from "react";
import type { Plan, Run, SSEEvent, WorkspaceEntry, WorkspaceFile } from "../api/types";
import { WorkspaceExplorer } from "./WorkspaceExplorer";
import { stageFromEventType, milestoneStatusFromEvents, statusColor } from "./stageUtils";
import {
  formatRoleLabel,
  formatTokenCount,
  summarizeSessionUsage,
} from "./sessionTokens";

interface Props {
  plan: Plan | null;
  events: SSEEvent[];
  sessionStatus: string;
  activeRun: Run | null;
  tree: WorkspaceEntry | null;
  file: WorkspaceFile | null;
  fileLoading: boolean;
  onOpenFile: (path: string) => void;
  onRefreshWorkspace: () => void;
  onCancelRun: () => Promise<void>;
}

function eventTime(ev: SSEEvent): string {
  return ev.ts.split("T")[1] || ev.ts;
}

function eventSummary(ev: SSEEvent): string {
  const data = ev.data || {};
  if (ev.type === "tool.called") {
    const tool = typeof data.tool === "string" ? data.tool : "tool";
    const reasoning = typeof data.reasoning === "string" ? data.reasoning : "";
    return reasoning ? `${tool}: ${reasoning}` : tool;
  }
  if (ev.type === "validation.contract_run") {
    return `contract exited ${String(data.returncode ?? "?")}`;
  }
  if (ev.type === "validation.finished") {
    return `verdict ${String(data.verdict ?? "?")}`;
  }
  if (ev.type === "milestone.replan") {
    return typeof data.guidance === "string" ? data.guidance : "replan requested";
  }
  if (ev.type === "milestone.passed") {
    const files = Array.isArray(data.files_modified) ? data.files_modified.length : 0;
    return files ? `${files} file(s) modified` : "passed";
  }
  if (ev.type === "worker.invalid_json") {
    return `attempt ${String(data.attempt ?? "?")} / ${String(data.max_attempts ?? "?")}`;
  }
  if (typeof data.title === "string") return data.title;
  if (typeof data.reason === "string") return data.reason;
  if (typeof data.status === "string") return data.status;
  return "";
}

function eventGroup(events: SSEEvent[], limit = 8) {
  return events.slice(-limit).reverse();
}

export function RunInspector({
  plan,
  events,
  sessionStatus,
  activeRun,
  tree,
  file,
  fileLoading,
  onOpenFile,
  onRefreshWorkspace,
  onCancelRun,
}: Props) {
  const [openMilestones, setOpenMilestones] = useState<Set<string>>(new Set());
  const [agentsOpen, setAgentsOpen] = useState(false);
  const [openAgents, setOpenAgents] = useState<Set<string>>(new Set());
  const [cancelling, setCancelling] = useState(false);

  const currentStage = useMemo(() => {
    if (events.length === 0) return "Idle";
    return stageFromEventType(events[events.length - 1].type);
  }, [events]);

  const milestoneStatuses = useMemo(() => milestoneStatusFromEvents(events), [events]);

  const eventsByMilestone = useMemo(() => {
    const groups: Record<string, SSEEvent[]> = {};
    for (const ev of events) {
      const msId = ev.data?.milestone_id;
      const key = typeof msId === "string" ? msId : "__mission__";
      groups[key] = groups[key] || [];
      groups[key].push(ev);
    }
    return groups;
  }, [events]);

  const milestones = plan?.milestones || [];
  const passedCount = Object.values(milestoneStatuses).filter((s) => s === "passed").length;
  const tokenTotals = useMemo(() => summarizeSessionUsage(events), [events]);

  const toggleMilestone = (msId: string) => {
    setOpenMilestones((prev) => {
      const next = new Set(prev);
      if (next.has(msId)) {
        next.delete(msId);
      } else {
        next.add(msId);
      }
      return next;
    });
  };

  const toggleAgent = (role: string) => {
    setOpenAgents((prev) => {
      const next = new Set(prev);
      if (next.has(role)) next.delete(role);
      else next.add(role);
      return next;
    });
  };

  return (
    <div className="run-inspector">
      <section className="inspector-section">
        <div className="panel-header">
          <span>Run Inspector</span>
          <span className="panel-actions">
            {activeRun && (
              <button
                className="danger"
                disabled={cancelling}
                onClick={async () => {
                  setCancelling(true);
                  try {
                    await onCancelRun();
                  } finally {
                    setCancelling(false);
                  }
                }}
              >
                {cancelling ? "Stopping..." : "Stop"}
              </button>
            )}
            <span className={`status-badge ${statusColor(sessionStatus)}`}>{sessionStatus}</span>
          </span>
        </div>

        <div className="stage-indicator">
          <div className="stage-label">Current Stage</div>
          <div className="stage-value">{currentStage}</div>
          <div className="stage-meta">
            {milestones.length > 0
              ? `${passedCount} / ${milestones.length} milestones passed`
              : "Waiting for plan"}
          </div>
          {(tokenTotals.calls > 0 || tokenTotals.byRole.length > 0) && (
            <div className="session-token-totals">
              <div className="stage-label">Session tokens</div>
              <div className="session-token-line">
                {formatTokenCount(tokenTotals.prompt)} prefill
                <span className="session-token-sep">·</span>
                {formatTokenCount(tokenTotals.generated)} generated
                {tokenTotals.estimated ? <span className="session-token-est">est.</span> : null}
              </div>
              <div className="session-token-line muted">
                {formatTokenCount(tokenTotals.total)} total
                <span className="session-token-sep">·</span>
                {tokenTotals.calls} LLM {tokenTotals.calls === 1 ? "call" : "calls"}
              </div>
              {tokenTotals.byRole.length > 0 && (
                <div className="agent-usage">
                  <button
                    className="agent-usage-toggle"
                    onClick={() => setAgentsOpen((open) => !open)}
                  >
                    <span className="ms-expand">{agentsOpen ? "-" : "+"}</span>
                    By agent
                  </button>
                  {agentsOpen && (
                    <div className="agent-usage-list">
                      {tokenTotals.byRole.map((agent) => {
                        const open = openAgents.has(agent.role);
                        return (
                          <div key={agent.role} className="agent-usage-item">
                            <button
                              className="agent-usage-row"
                              onClick={() => toggleAgent(agent.role)}
                            >
                              <span className="ms-expand">{open ? "-" : "+"}</span>
                              <span className="agent-usage-name">{formatRoleLabel(agent.role)}</span>
                              <span className="agent-usage-summary">
                                {formatTokenCount(agent.total)} tok
                                <span className="session-token-sep">·</span>
                                {agent.toolCalls} tool{agent.toolCalls === 1 ? "" : "s"}
                              </span>
                            </button>
                            {open && (
                              <div className="agent-usage-detail">
                                <div>
                                  {formatTokenCount(agent.prompt)} prefill
                                  <span className="session-token-sep">·</span>
                                  {formatTokenCount(agent.generated)} generated
                                  <span className="session-token-sep">·</span>
                                  {agent.calls} LLM {agent.calls === 1 ? "call" : "calls"}
                                  {agent.estimated ? <span className="session-token-est">est.</span> : null}
                                </div>
                                {agent.tools.length === 0 ? (
                                  <div className="empty-inline">No tools.</div>
                                ) : (
                                  <ul className="agent-tool-list">
                                    {agent.tools.map((item) => (
                                      <li key={item.tool}>
                                        <span>{item.tool}</span>
                                        <span>×{item.count}</span>
                                      </li>
                                    ))}
                                  </ul>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="mission-events">
          <div className="subsection-title">Mission Events</div>
          {eventGroup(eventsByMilestone.__mission__ || [], 5).map((ev, i) => (
            <div className="event-row compact" key={`${ev.ts}-${ev.type}-${i}`}>
              <span className="ev-time">{eventTime(ev)}</span>
              <span className="ev-type">{ev.type}</span>
              <span className="ev-summary">{eventSummary(ev)}</span>
            </div>
          ))}
        </div>

        <div className="milestone-list inspector-milestones">
          {milestones.length === 0 ? (
            <div className="empty-state small-empty">No plan yet.</div>
          ) : (
            milestones.map((ms, i) => {
              const msId = ms.id || `M${i + 1}`;
              const status = milestoneStatuses[msId] || ms.status || "pending";
              const msEvents = eventsByMilestone[msId] || [];
              const open = openMilestones.has(msId) || status === "running";
              return (
                <div key={msId} className={`milestone-item ${open ? "open" : ""}`}>
                  <button className="milestone-button" onClick={() => toggleMilestone(msId)}>
                    <span className="ms-expand">{open ? "-" : "+"}</span>
                    <span className="ms-copy">
                      <span className="ms-header">
                        <span className="ms-id">{msId}</span>
                        <span className={`status-badge ${statusColor(status)}`}>{status}</span>
                      </span>
                      {ms.title && <span className="ms-title">{ms.title}</span>}
                    </span>
                    <span className="ms-event-count">{msEvents.length}</span>
                  </button>
                  {open && (
                    <div className="milestone-events">
                      {msEvents.length === 0 ? (
                        <div className="empty-inline">No events for this milestone yet.</div>
                      ) : (
                        eventGroup(msEvents, 20).map((ev, idx) => (
                          <div className="event-row compact" key={`${ev.ts}-${ev.type}-${idx}`}>
                            <span className="ev-time">{eventTime(ev)}</span>
                            <span className="ev-type">{ev.type}</span>
                            <span className="ev-summary">{eventSummary(ev)}</span>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </section>

      <section className="inspector-section workspace-section">
        <WorkspaceExplorer
          tree={tree}
          file={file}
          fileLoading={fileLoading}
          onOpenFile={onOpenFile}
          onRefresh={onRefreshWorkspace}
        />
      </section>
    </div>
  );
}
