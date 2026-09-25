import type { DecisionAction, PendingDecision } from "../api/types";

interface Props {
  decision: PendingDecision;
  resolving: boolean;
  onResolve: (action: DecisionAction) => void;
}

const ACTION_LABELS: Record<DecisionAction, string> = {
  apply_fix: "Apply fix",
  run_smoke: "Run smoke checks",
  setup_env: "Set up test env",
  escalate_mission: "Escalate to mission",
  dismiss: "Dismiss",
};

const ACTION_CLASS: Record<DecisionAction, string> = {
  apply_fix: "primary",
  run_smoke: "primary",
  setup_env: "primary",
  escalate_mission: "",
  dismiss: "",
};

export function DecisionBanner({ decision, resolving, onResolve }: Props) {
  return (
    <div className="decision-banner">
      <div className="decision-banner-title">
        <span className="decision-dot" />
        {decision.title}
      </div>
      <pre className="decision-banner-summary">{decision.summary}</pre>
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
    </div>
  );
}
