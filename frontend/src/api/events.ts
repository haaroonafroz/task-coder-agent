// SSE helper: open an EventSource to /api/v1/sessions/{sid}/events
// and invoke onEvent for each parsed event. Returns a cleanup function.

import type { SSEEvent } from "./types";

export function openEventStream(
  sid: string,
  onEvent: (event: SSEEvent) => void,
  onError?: (err: Event) => void
): () => void {
  const url = `/api/v1/sessions/${sid}/events`;
  const source = new EventSource(url);

  source.onmessage = (ev: MessageEvent) => {
    try {
      const parsed: SSEEvent = JSON.parse(ev.data);
      onEvent(parsed);
    } catch {
      // ignore non-JSON lines
    }
  };

  // Also listen for named events (the server sends event: <type> lines).
  // EventSource fires onmessage for unnamed events; for named events we
  // need explicit listeners. We handle the common ones generically.
  const handler = (ev: MessageEvent) => {
    try {
      const parsed: SSEEvent = JSON.parse(ev.data);
      onEvent(parsed);
    } catch {
      // ignore
    }
  };

  const eventTypes = [
    "session.started",
    "plan.created",
    "plan.updated",
    "milestone.started",
    "milestone.skipped",
    "milestone.passed",
    "milestone.failed",
    "milestone.replan",
    "milestone.blocked",
    "milestone.retry",
    "milestone.retries_exhausted",
    "worker.started",
    "worker.invalid_json",
    "worker.complete",
    "worker.complete_rejected",
    "worker.blocked",
    "tool.called",
    "tool.result",
    "validation.started",
    "validation.contract_run",
    "validation.spec_gaming",
    "validation.finished",
    "handoff.saved",
    "mission.complete",
  ];
  eventTypes.forEach((t) => source.addEventListener(t, handler));

  if (onError) {
    source.onerror = (err: Event) => {
      onError(err);
    };
  }

  return () => {
    source.close();
  };
}
