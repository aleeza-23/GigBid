"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { api, type ApprovalResult, type PendingApproval } from "@/lib/api/client";
import { CoverageBadge, RiskBadge } from "@/components/Badges";
import { EvidenceHover } from "@/components/EvidenceHover";
import { ErrorState, Loading } from "@/components/States";

/**
 * The proposal studio. This is where the human is actually in control.
 *
 * Everything C5 requires on one screen: the draft with every claim's evidence on hover,
 * the Critic's and Compliance's comments inline, the risk report, the price rationale,
 * and the dry-run preview of exactly what would be sent and what it costs. Then approve,
 * edit, or reject with a reason.
 *
 * Two deliberate choices:
 *
 * - **the cost is stated next to the button**, not buried in a panel. Approving spends
 *   Connects and cannot be undone, so the number belongs where the decision is made;
 * - **editing switches to a plain textarea**, losing the hover highlighting. Keeping the
 *   annotations during editing would mean reconciling claim offsets against every
 *   keystroke, and getting that subtly wrong would show the wrong evidence under a
 *   sentence — worse than showing none.
 */
export default function StudioPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const threadId = decodeURIComponent(params.id);

  const [approval, setApproval] = useState<PendingApproval | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [result, setResult] = useState<ApprovalResult | null>(null);

  const load = useCallback(() => {
    setError(null);
    api
      .approval(threadId)
      .then((item) => {
        setApproval(item);
        setDraft(String(item.payload?.draft?.cover_letter ?? ""));
      })
      .catch(setError);
  }, [threadId]);

  useEffect(load, [load]);

  async function decide(action: "approve" | "edit" | "reject") {
    setBusy(action);
    try {
      const body =
        action === "edit"
          ? { action, cover_letter: draft, text: draft, reason: reason || undefined }
          : { action, reason: reason || undefined };
      setResult(await api.decide(threadId, body));
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  }

  if (error && !approval) return <ErrorState error={error} retry={load} />;

  if (result) {
    const sent = result.status === "submitted";
    return (
      <>
        <h1>{sent ? "Submitted" : result.status.replace(/_/g, " ")}</h1>
        <div className={`banner ${sent ? "banner-warn" : ""}`} role="status">
          {sent ? (
            <>
              <strong>Sent to the marketplace.</strong>{" "}
              {result.proposal_id && (
                <>
                  Proposal <span className="mono">{result.proposal_id}</span>.{" "}
                </>
              )}
              {result.connects_spent ? `${result.connects_spent} Connects spent.` : ""}
              {result.idempotent_replay && " (This was already submitted; nothing was spent again.)"}
            </>
          ) : result.status === "rejected" ? (
            <>Rejected. Nothing was sent and no Connects were spent.</>
          ) : (
            <>Ended as {result.status}. Nothing was sent.</>
          )}
        </div>
        {result.errors.length > 0 && (
          <div className="banner banner-danger" role="alert">
            <ul style={{ paddingLeft: 18, margin: 0 }}>
              {result.errors.map((err, index) => (
                <li key={index}>{String(err.error)}</li>
              ))}
            </ul>
          </div>
        )}
        <div className="btn-row">
          <Link className="btn btn-primary" href="/inbox">
            Back to inbox
          </Link>
          <Link className="btn" href="/pipeline">
            Pipeline
          </Link>
        </div>
      </>
    );
  }

  if (!approval) return <Loading what="Loading the proposal" />;

  const payload = approval.payload;
  const preview = payload.dry_run_preview ?? {};
  const compliance = payload.compliance ?? {};
  const critique = payload.critique ?? {};
  const verification = payload.verification ?? {};
  const claims = payload.claims_with_evidence ?? [];
  const connectsCost = preview.connects_cost;

  return (
    <>
      <p className="small">
        <Link href="/inbox">← Inbox</Link>
        {payload.job?.job_id && (
          <>
            {" · "}
            <Link href={`/jobs/${encodeURIComponent(payload.job.job_id)}`}>
              {payload.job.job_id}
            </Link>
          </>
        )}
        {" · "}
        <Link href={`/runs/${encodeURIComponent(threadId)}`}>run trace</Link>
      </p>

      <h1>{payload.job?.title ?? "Proposal"}</h1>
      <p className="subtitle">
        Nothing has been sent. Approving spends Connects and cannot be undone.
      </p>

      {error && <ErrorState error={error} />}

      <div className="grid grid-3" style={{ marginBottom: 14 }}>
        <div className="stat">
          <div className="stat-value">
            {payload.fit?.coverage !== null && payload.fit?.coverage !== undefined
              ? `${Math.round(payload.fit.coverage * 100)}%`
              : "—"}
          </div>
          <div className="stat-label">Requirements covered</div>
        </div>
        <div className="stat">
          <div className="stat-value">
            {payload.price?.amount_usd ? `$${Number(payload.price.amount_usd).toFixed(0)}` : "—"}
          </div>
          <div className="stat-label">Bid {payload.price?.type ?? ""}</div>
        </div>
        <div className="stat">
          <div className="stat-value">{connectsCost ?? "—"}</div>
          <div className="stat-label">Connects it costs</div>
        </div>
        <div className="stat">
          <div className="stat-value">{verification.unsupported_after ?? "—"}</div>
          <div className="stat-label">Unsupported claims left</div>
        </div>
      </div>

      {critique.unresolved && (
        <div className="banner banner-warn" role="status">
          <strong>The Critic still objects.</strong> It asked for changes twice, which is
          the limit, so the draft came through anyway.{" "}
          {(critique.reasons ?? []).join("; ")}
        </div>
      )}

      {compliance.status === "fixed" && (
        <div className="banner banner-warn" role="status">
          <strong>Compliance edited this draft.</strong>{" "}
          {(compliance.redactions ?? []).join("; ")}
        </div>
      )}

      {(payload.draft?.unsupported_claims_removed ?? []).length > 0 && (
        <div className="banner banner-warn" role="status">
          <strong>Claims removed as unsupported:</strong>
          <ul className="small" style={{ paddingLeft: 18, margin: "6px 0 0" }}>
            {payload.draft.unsupported_claims_removed.map((claim: string) => (
              <li key={claim}>{claim}</li>
            ))}
          </ul>
        </div>
      )}

      {(preview.warnings ?? []).length > 0 && (
        <div className="banner banner-danger" role="alert">
          <strong>Before you send:</strong> {(preview.warnings ?? []).join("; ")}
        </div>
      )}

      <section className="card">
        <div className="card-row">
          <h2 style={{ margin: 0 }}>The draft</h2>
          <div className="btn-row">
            <RiskBadge score={payload.risk?.score} />
            <CoverageBadge coverage={payload.fit?.coverage} />
            <button
              style={{ minHeight: 32, padding: "4px 12px" }}
              onClick={() => setEditing((value) => !value)}
            >
              {editing ? "Done editing" : "Edit"}
            </button>
          </div>
        </div>

        {editing ? (
          <>
            <label htmlFor="draft">Proposal text</label>
            <textarea
              id="draft"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              aria-describedby="edit-note"
            />
            <p id="edit-note" className="small dim">
              Your edit is checked against the policy pack again before it is sent, and the
              difference is recorded so GigBid can learn your style. It needs the same
              change three times before it becomes a rule.
            </p>
          </>
        ) : (
          <>
            <EvidenceHover text={draft} claims={claims} />
            <p className="small dim" style={{ marginTop: 12 }}>
              Underlined sentences have evidence behind them. Hover or tab to a sentence to
              see it.
            </p>
          </>
        )}
      </section>

      <div className="grid grid-2" style={{ marginTop: 12 }}>
        <section className="card">
          <h3 style={{ marginTop: 0 }}>Why this price</h3>
          <p className="small">{payload.price?.rationale ?? "—"}</p>
          <p className="small dim">
            Floor ${payload.price?.profile_floor_usd ?? "—"}
            {payload.price?.above_floor === false && (
              <strong style={{ color: "var(--danger)" }}> · below the floor</strong>
            )}
          </p>
        </section>

        <section className="card">
          <h3 style={{ marginTop: 0 }}>Risk</h3>
          <p className="small">{payload.risk?.explanation ?? "—"}</p>
          {(payload.risk?.flags ?? []).length > 0 && (
            <ul className="small" style={{ paddingLeft: 18, marginBottom: 0 }}>
              {payload.risk.flags.map((flag: string) => (
                <li key={flag}>{flag}</li>
              ))}
            </ul>
          )}
        </section>
      </div>

      {(payload.fit?.missing ?? []).length > 0 && (
        <section className="card" style={{ marginTop: 12 }}>
          <h3 style={{ marginTop: 0 }}>Requirements with no evidence</h3>
          <p className="small dim">The proposal must not claim these.</p>
          <p className="small">
            {payload.fit.missing.map((item: string) => (
              <span key={item} className="pill pill-danger" style={{ marginRight: 4 }}>
                {item}
              </span>
            ))}
          </p>
        </section>
      )}

      <section className="card" style={{ marginTop: 12 }}>
        <h3 style={{ marginTop: 0 }}>Exactly what will be sent</h3>
        <pre
          className="mono small"
          style={{
            background: "var(--surface-2)",
            padding: 12,
            borderRadius: 8,
            overflowX: "auto",
            margin: 0,
          }}
        >
          {JSON.stringify(preview.would_send ?? {}, null, 2)}
        </pre>
        <p className="small dim">
          Cost {connectsCost ?? "?"} Connects · balance after{" "}
          {preview.connects_balance_after ?? "?"}
          {preview.within_weekly_budget === false && (
            <strong style={{ color: "var(--danger)" }}> · over the weekly budget</strong>
          )}
        </p>
      </section>

      <section className="card" style={{ marginTop: 12 }}>
        <div className="field">
          <label htmlFor="reason">Reason (optional, but useful if you reject)</label>
          <input
            id="reason"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="e.g. rate too low for the scope"
            style={{ minHeight: 40 }}
          />
        </div>
        <div className="btn-row">
          <button
            className="btn-primary"
            onClick={() => decide(editing ? "edit" : "approve")}
            disabled={busy !== null || (editing && draft.trim().length === 0)}
          >
            {busy === "approve" || busy === "edit"
              ? "Sending…"
              : editing
                ? `Send my edited version${connectsCost ? ` · ${connectsCost} Connects` : ""}`
                : `Approve and send${connectsCost ? ` · ${connectsCost} Connects` : ""}`}
          </button>
          <button className="btn-danger" onClick={() => decide("reject")} disabled={busy !== null}>
            {busy === "reject" ? "Rejecting…" : "Reject"}
          </button>
        </div>
        <p className="small dim" style={{ marginBottom: 0 }}>
          {payload.warning}
        </p>
      </section>
    </>
  );
}
