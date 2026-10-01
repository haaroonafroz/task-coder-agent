import type { DecisionAction, PendingDecision } from "../api/types";
import { Markdown } from "./Markdown";

interface Props {
  decision: PendingDecision;
  resolving: boolean;
  onResolve: (action: DecisionAction) => void;
}

const ACTION_LABELS: Record<string, string> = {
  apply_fix: "Apply fix",
  run_smoke: "Run smoke checks",
  setup_env: "Set up test env",
  escalate_mission: "Escalate to mission",
  dismiss: "Dismiss",
  approve_plan: "Build",
};

const ACTION_CLASS: Record<string, string> = {
  apply_fix: "primary",
  run_smoke: "primary",
  setup_env: "primary",
  escalate_mission: "",
  dismiss: "",
  approve_plan: "primary",
};

export function DecisionBanner({ decision, resolving, onResolve }: Props) {
  const isPlan = decision.type === "plan_approval";

  return (
    <div className={`decision-banner ${isPlan ? "plan-card" : ""}`}>
      <div className="decision-banner-title">
        <span className="decision-dot" />
        {isPlan ? `Plan: ${decision.title}` : decision.title}
      </div>
      <div className="decision-banner-summary">
        <Markdown text={decision.summary} />
      </div>
      <div className="decision-banner-actions">
        {decision.options.map((action) => (
          <button
            key={action}
            className={ACTION_CLASS[action] || undefined}
            disabled={resolving}
            onClick={() => onResolve(action)}
          >
            {ACTION_LABELS[action] || action}
          </button>
        ))}
      </div>
      {resolving && <div className="decision-banner-note">Starting follow-up…</div>}
      {isPlan && !resolving && (
        <div className="decision-banner-note">
          Build runs the draft. Send a message in Plan mode to adjust it.
        </div>
      )}
    </div>
  );
}

