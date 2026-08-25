import { useState } from "react";
import type { Session, ModelChoice } from "../api/types";
import { api } from "../api/client";
import { statusColor } from "./stageUtils";
import { ModelSelect } from "./ModelSelect";
import { useModels } from "../hooks";

interface Props {
  sessions: Session[];
  activeSid: string | null;
  onSelect: (sid: string) => void;
  onCreated: () => void;
}

export function SessionSidebar({ sessions, activeSid, onSelect, onCreated }: Props) {
  const [showForm, setShowForm] = useState(false);
  const [title, setTitle] = useState("");
  const [model, setModel] = useState<ModelChoice>("auto");
  const [creating, setCreating] = useState(false);
  const { models } = useModels();

  const handleCreate = async () => {
    if (!title.trim()) return;
    setCreating(true);
    try {
      const s = await api.createSession({ title: title.trim(), model });
      setTitle("");
      setShowForm(false);
      onCreated();
      onSelect(s.session_id);
    } catch (e) {
      alert(`Failed to create session: ${e}`);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="panel" style={{ overflow: "hidden" }}>
      <div className="panel-header">
        <span>Sessions</span>
        <button
          style={{ padding: "2px 8px", fontSize: 12 }}
          onClick={() => setShowForm(!showForm)}
        >
          {showForm ? "Cancel" : "+ New"}
        </button>
      </div>

      {showForm && (
        <div className="new-session-form">
          <input
            type="text"
            placeholder="Session title..."
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            autoFocus
          />
          <ModelSelect value={model} models={models} onChange={setModel} />
          <button className="primary" disabled={creating || !title.trim()} onClick={handleCreate}>
            {creating ? "Creating..." : "Create"}
          </button>
        </div>
      )}

      <div className="panel-body">
        {sessions.length === 0 && (
          <div className="empty-state" style={{ fontSize: 12 }}>
            No sessions yet.
          </div>
        )}
        {sessions.map((s) => (
          <div
            key={s.session_id}
            className={`session-item ${activeSid === s.session_id ? "active" : ""}`}
            onClick={() => onSelect(s.session_id)}
          >
            <div className="title">{s.title}</div>
            <div className="meta">
              <span className={`status-badge ${statusColor(s.status)}`}>{s.status}</span>
              <span>{s.selected_model}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
