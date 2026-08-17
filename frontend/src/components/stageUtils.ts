// Derive the current stage label from the latest event type.

export function stageFromEventType(eventType: string): string {
  const map: Record<string, string> = {
    "session.started": "Starting",
    "plan.created": "Orchestration",
    "orchestrator.explore.started": "Orchestrator Exploring",
    "orchestrator.explore.finished": "Orchestration",
    "plan.updated": "Replanning",
    "milestone.started": "Milestone Started",
    "milestone.skipped": "Milestone Skipped",
    "worker.started": "Worker",
    "tool.called": "Tool Execution",
    "tool.result": "Tool Execution",
    "validation.started": "Validation",
    "validation.contract_run": "Validation",
    "validation.spec_gaming": "Spec Gaming Detected",
    "validation.finished": "Validation Complete",
    "milestone.retry": "Retrying",
    "milestone.replan": "Replanning",
    "milestone.passed": "Milestone Passed",
    "milestone.failed": "Milestone Failed",
    "milestone.blocked": "Blocked",
    "milestone.retries_exhausted": "Retries Exhausted",
    "handoff.saved": "Handoff Saved",
    "mission.complete": "Mission Complete",
    "mission.cancelled": "Cancelled",
  };
  return map[eventType] || eventType;
}

export function statusColor(status: string): string {
  const colors: Record<string, string> = {
    created: "status-created",
    running: "status-running",
    completed: "status-completed",
    partial: "status-partial",
    failed: "status-failed",
    queued: "status-queued",
    error: "status-error",
    cancelled: "status-failed",
    planning: "status-planning",
    paused: "status-created",
    pending: "status-created",
    passed: "status-completed",
    blocked: "status-failed",
    replan: "status-planning",
    skipped: "status-partial",
  };
  return colors[status] || "status-created";
}

// Determine which milestones are done based on events + plan.
export function milestoneStatusFromEvents(
  events: { type: string; data: Record<string, unknown> }[]
): Record<string, string> {
  const statuses: Record<string, string> = {};
  for (const ev of events) {
    const msId = ev.data?.milestone_id as string | undefined;
    if (!msId) continue;
    switch (ev.type) {
      case "milestone.started":
        if (!statuses[msId] || statuses[msId] === "pending")
          statuses[msId] = "running";
        break;
      case "milestone.skipped":
        statuses[msId] = "skipped";
        break;
      case "milestone.passed":
        statuses[msId] = "passed";
        break;
      case "milestone.failed":
        statuses[msId] = "failed";
        break;
      case "milestone.blocked":
        statuses[msId] = "blocked";
        break;
      case "milestone.replan":
        statuses[msId] = "replan";
        break;
    }
  }
  return statuses;
}
