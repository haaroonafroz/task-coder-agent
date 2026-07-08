import { useMemo } from "react";
import type { SSEEvent } from "../api/types";

interface Props {
  events: SSEEvent[];
}

export function EventDrawer({ events }: Props) {
  const recent = useMemo(() => events.slice(-100).reverse(), [events]);

  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      <div className="panel-header">
        <span>Events ({events.length})</span>
      </div>
      <div className="panel-body event-list">
        {recent.length === 0 ? (
          <div className="empty-state" style={{ fontSize: 12 }}>
            No events yet.
          </div>
        ) : (
          recent.map((ev, i) => (
            <div key={i} className="event-row">
              <span className="ev-time">{ev.ts.split("T")[1] || ev.ts}</span>
              <span className="ev-type">{ev.type}</span>
              <span style={{ color: "var(--text-muted)" }}>
                {ev.data?.milestone_id ? `[${ev.data.milestone_id}]` : ""}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
