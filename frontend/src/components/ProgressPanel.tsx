import { useMemo } from "react";
import type { Plan, SSEEvent } from "../api/types";
import { stageFromEventType, milestoneStatusFromEvents, statusColor } from "./stageUtils";

interface Props {
  plan: Plan | null;
  events: SSEEvent[];
  sessionStatus: string;
}

export function ProgressPanel({ plan, events, sessionStatus }: Props) {
  const currentStage = useMemo(() => {
    if (events.length === 0) return "Idle";
    const lastEv = events[events.length - 1];
    return stageFromEventType(lastEv.type);
  }, [events]);

  const milestoneStatuses = useMemo(
    () => milestoneStatusFromEvents(events),
    [events]
  );

  const passedCount = useMemo(
    () => Object.values(milestoneStatuses).filter((s) => s === "passed").length,
    [milestoneStatuses]
  );

  const totalCount = plan?.milestones?.length || 0;

  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      <div className="panel-header">
        <span>Progress</span>
        <span className={`status-badge ${statusColor(sessionStatus)}`}>{sessionStatus}</span>
      </div>

      <div className="stage-indicator">
        <div className="stage-label">Current Stage</div>
        <div className="stage-value">{currentStage}</div>
        {totalCount > 0 && (
          <div style={{ marginTop: 6, fontSize: 13, color: "var(--text-secondary)" }}>
            Milestones: {passedCount} / {totalCount} passed
          </div>
        )}
      </div>

      <div className="panel-body">
        {plan && plan.milestones && plan.milestones.length > 0 ? (
          <div className="milestone-list">
            {plan.milestones.map((ms, i) => {
              const msId = ms.id || `M${i + 1}`;
              const status = milestoneStatuses[msId] || ms.status || "pending";
              const isActive = status === "running";
              return (
                <div
                  key={msId}
                  className="milestone-item"
                  style={{
                    borderLeft: isActive
                      ? "3px solid var(--accent)"
                      : status === "passed"
                      ? "3px solid var(--success)"
                      : status === "failed"
                      ? "3px solid var(--danger)"
                      : status === "blocked"
                      ? "3px solid var(--warning)"
                      : "3px solid transparent",
                  }}
                >
                  <div className="ms-header">
                    <span className="ms-id">{msId}</span>
                    <span className={`status-badge ${statusColor(status)}`}>{status}</span>
                  </div>
                  {ms.title && <div className="ms-title">{ms.title}</div>}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="empty-state" style={{ fontSize: 12 }}>
            No plan yet.
          </div>
        )}
      </div>
    </div>
  );
}
