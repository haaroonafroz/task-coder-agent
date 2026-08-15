import { useEffect, useMemo, useRef, useState } from "react";
import type { Session, Message, ModelChoice, SSEEvent } from "../api/types";
import { ModelSelect } from "./ModelSelect";
import { PersonaTurn } from "./PersonaTurn";
import { useModels } from "../hooks";
import { buildChatItems, formatTs, missionStatusLabel } from "../hooks/useChatTurns";

interface Props {
  session: Session | null;
  messages: Message[];
  sending: boolean;
  connected: boolean;
  events: SSEEvent[];
  onSend: (content: string, triggerRun: boolean, model?: string) => void;
}

export function ChatPanel({ session, messages, sending, connected, events, onSend }: Props) {
  const [input, setInput] = useState("");
  const [model, setModel] = useState<ModelChoice>("auto");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { models } = useModels();

  const contextLength = useMemo(() => {
    const local = models.find((entry) => entry.key === "local");
    const auto = models.find((entry) => entry.key === "auto");
    return local?.context_length ?? auto?.context_length ?? null;
  }, [models]);

  const feed = useMemo(
    () => buildChatItems(messages, events),
    [messages, events],
  );

  useEffect(() => {
    if (session) {
      setModel((session.selected_model as ModelChoice) || "auto");
    }
  }, [session]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [feed]);

  const handleSend = () => {
    if (!input.trim() || sending) return;
    onSend(input.trim(), true, model);
    setInput("");
  };

  if (!session) {
    return (
      <div className="panel">
        <div className="empty-state">Select or create a session to start chatting.</div>
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
        {feed.length === 0 && (
          <div className="empty-state" style={{ fontSize: 12 }}>
            No messages yet. Send a request below to start a mission.
          </div>
        )}
        {feed.map((item) => {
          if (item.kind === "user") {
            return (
              <div key={item.id} className="message user">
                <div className="role">You</div>
                <div className="content">{item.content}</div>
                <div className="ts">{formatTs(item.ts)}</div>
              </div>
            );
          }
          if (item.kind === "system") {
            return (
              <div key={item.id} className="message system-message">
                <div className="content">{item.content}</div>
                <div className="ts">{formatTs(item.ts)}</div>
              </div>
            );
          }
          if (item.kind === "mission_summary") {
            return (
              <div key={item.id} className={`message mission-summary mission-${item.status}`}>
                <div className="role persona-role">
                  <span>Mission recap</span>
                  <span className={`mission-status-badge status-${item.status}`}>
                    {missionStatusLabel(item.status)}
                  </span>
                </div>
                <pre className="mission-summary-body">{item.content}</pre>
                <div className="ts">{formatTs(item.ts)}</div>
              </div>
            );
          }
          return (
            <PersonaTurn
              key={item.id}
              turn={item}
              contextLength={contextLength}
            />
          );
        })}
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
