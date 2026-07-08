import { useEffect, useRef, useState } from "react";
import type { Session, Message, ModelChoice } from "../api/types";
import { ModelSelect } from "./ModelSelect";
import { useModels } from "../hooks";

interface Props {
  session: Session | null;
  messages: Message[];
  sending: boolean;
  connected: boolean;
  onSend: (content: string, triggerRun: boolean, model?: string) => void;
}

export function ChatPanel({ session, messages, sending, connected, onSend }: Props) {
  const [input, setInput] = useState("");
  const [model, setModel] = useState<ModelChoice>("auto");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { models } = useModels();

  useEffect(() => {
    if (session) {
      setModel((session.selected_model as ModelChoice) || "auto");
    }
  }, [session]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim() || sending) return;
    onSend(input.trim(), true, model);
    setInput("");
  };

  if (!session) {
    return (
      <div className="panel">
        <div className="empty-state">
          Select or create a session to start chatting.
        </div>
      </div>
    );
  }

  return (
    <div className="panel chat-container">
      <div className="panel-header">
        <span>{session.title}</span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className={`conn-dot ${connected ? "connected" : "disconnected"}`} />
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            {connected ? "Live" : "Offline"}
          </span>
        </span>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state" style={{ fontSize: 12 }}>
            No messages yet. Send a request below to start a mission.
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <div className="role">
              {msg.role}
              {msg.run_id && (
                <span style={{ color: "var(--accent)", marginLeft: 6 }}>
                  run:{msg.run_id.slice(0, 8)}
                </span>
              )}
            </div>
            <div className="content">{msg.content}</div>
            <div className="ts">{msg.ts}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-composer">
        <div className="composer-row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Describe what to build or fix..."
            rows={2}
          />
        </div>
        <div className="composer-row" style={{ justifyContent: "space-between" }}>
          <ModelSelect
            className="model-select"
            value={model}
            models={models}
            onChange={setModel}
          />
          <button
            className="primary"
            disabled={sending || !input.trim()}
            onClick={handleSend}
          >
            {sending ? "Sending..." : "Send & Run"}
          </button>
        </div>
      </div>
    </div>
  );
}
