/**
 * Status and risk badges.
 *
 * Every badge carries a text label beside its colour. Colour alone would fail anyone with
 * a colour vision deficiency, and "risk 0.82" is more useful than a red dot regardless.
 */

export function RiskBadge({ score }: { score?: number | null }) {
  if (score === null || score === undefined) {
    return <span className="pill">risk not assessed</span>;
  }
  const tone = score >= 0.7 ? "pill-danger" : score >= 0.35 ? "pill-warn" : "pill-ok";
  const label = score >= 0.7 ? "high risk" : score >= 0.35 ? "some risk" : "low risk";
  return (
    <span className={`pill ${tone}`}>
      {label} {score.toFixed(2)}
    </span>
  );
}

export function CoverageBadge({ coverage }: { coverage?: number | null }) {
  if (coverage === null || coverage === undefined) {
    return <span className="pill">not analysed</span>;
  }
  const pct = Math.round(coverage * 100);
  const tone = pct >= 60 ? "pill-ok" : pct >= 35 ? "pill-warn" : "pill-danger";
  return <span className={`pill ${tone}`}>{pct}% covered</span>;
}

export function ActionBadge({ action }: { action?: string | null }) {
  if (!action) return <span className="pill">no recommendation</span>;
  const tone =
    action === "bid" ? "pill-ok" : action === "watch" ? "pill-warn" : "pill-danger";
  return <span className={`pill ${tone}`}>{action}</span>;
}

export function StatusBadge({ status }: { status?: string | null }) {
  if (!status) return null;
  const tone =
    status === "awaiting_approval"
      ? "pill-warn"
      : status === "submitted"
        ? "pill-ok"
        : status === "needs_attention" || status === "dead_letter"
          ? "pill-danger"
          : "pill";
  return <span className={`pill ${tone}`}>{status.replace(/_/g, " ")}</span>;
}
