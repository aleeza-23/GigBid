"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type MemoryView, type Settings } from "@/lib/api/client";
import { ErrorState, Loading } from "@/components/States";

/**
 * Settings, and the memory viewer.
 *
 * The memory viewer is the part that matters. Part D requires the user to see and delete
 * what GigBid has learned, and viva question 6 asks for the promotion rule that let an
 * item in. So each learned item shows its evidence count and the rule that admitted it,
 * and deletion offers two strengths:
 *
 * - **Forget the rule** — stop applying it. The evidence stays, so a later promotion
 *   could re-derive it.
 * - **Forget the evidence too** — "this was never my preference", which cannot come back.
 *
 * Collapsing those two into one button would mean either a rule that keeps returning or
 * silently discarding the user's history.
 */
export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [memory, setMemory] = useState<MemoryView | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    api.settings().then(setSettings).catch(setError);
    api.memory().then(setMemory).catch(setError);
  }, []);

  useEffect(load, [load]);

  async function forget(kind: "style" | "outcome", key: string, evidence: boolean) {
    setBusy(key);
    try {
      await api.forget(kind, key, evidence);
      setMemory(await api.memory());
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  }

  if (error && !settings) return <ErrorState error={error} retry={load} />;
  if (!settings) return <Loading what="Loading settings" />;

  return (
    <>
      <h1>Settings</h1>
      <p className="subtitle">What is configured, and what GigBid has learned about you.</p>

      {error && <ErrorState error={error} />}

      {!settings.embeddings_spec_compliant && (
        <div className="banner banner-warn" role="status">
          <strong>Retrieval is on the development embedding backend.</strong> It is
          TF-IDF + SVD, not an open-weight model, so any retrieval number produced now is
          not reportable. Install <span className="mono">sentence-transformers</span> and
          set <span className="mono">retrieval.embeddings.backend: local</span>.
        </div>
      )}

      <h2>Configuration</h2>
      <div className="card" style={{ padding: 0 }}>
        <div className="scroll-x">
          <table>
            <tbody>
              <tr>
                <td>Persona</td>
                <td>{settings.persona}</td>
              </tr>
              <tr>
                <td>Autonomy</td>
                <td>
                  {settings.autonomy}
                  <span className="small dim"> · nothing is ever auto-sent</span>
                </td>
              </tr>
              <tr>
                <td>Hourly floor</td>
                <td>
                  {settings.hourly_rate_usd ? `$${settings.hourly_rate_usd}/hr` : "—"}
                  {settings.minimum_project_budget_usd
                    ? ` · min project $${settings.minimum_project_budget_usd}`
                    : ""}
                </td>
              </tr>
              <tr>
                <td>Weekly Connects budget</td>
                <td>
                  {settings.weekly_connects_budget ?? "—"}
                  {settings.connects_balance !== null && settings.connects_balance !== undefined
                    ? ` · ${settings.connects_balance} in the account`
                    : ""}
                </td>
              </tr>
              <tr>
                <td>Marketplace</td>
                <td>
                  {settings.marketplace_mode}
                  {settings.marketplace_mode === "upwork_live" && (
                    <span className="pill pill-warn" style={{ marginLeft: 6 }}>
                      live · writes are drafts you confirm on Upwork
                    </span>
                  )}
                </td>
              </tr>
              <tr>
                <td>Model profile</td>
                <td>{settings.model_profile}</td>
              </tr>
              <tr>
                <td>Retrieval</td>
                <td>
                  {settings.retrieval_variant} · embeddings {settings.embeddings_backend}
                  {settings.embeddings_spec_compliant ? (
                    <span className="pill pill-ok" style={{ marginLeft: 6 }}>
                      open-weight
                    </span>
                  ) : (
                    <span className="pill pill-warn" style={{ marginLeft: 6 }}>
                      development backend
                    </span>
                  )}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <h2>What GigBid has learned</h2>
      {!memory ? (
        <Loading what="Loading memory" />
      ) : (
        <>
          <p className="subtitle">
            {memory.style_rules.length + memory.outcome_rules.length === 0
              ? "Nothing promoted yet."
              : `${memory.style_rules.length} style rules, ${memory.outcome_rules.length} outcome rules.`}{" "}
            {memory.pending_style_observations} style observations and{" "}
            {memory.pending_outcome_observations} outcome observations are recorded but have
            not met the threshold.
          </p>

          <div className="card">
            <h3 style={{ marginTop: 0 }}>How something gets in</h3>
            <ul className="small" style={{ paddingLeft: 18, marginBottom: 0 }}>
              {memory.promotion_rules.map((rule: any, index: number) => (
                <li key={index} style={{ marginBottom: 6 }}>
                  <strong>{rule.kind}:</strong> {rule.rule}
                  <br />
                  <span className="dim">{rule.why} · inputs: {rule.inputs}</span>
                </li>
              ))}
            </ul>
          </div>

          {memory.style_rules.length > 0 && (
            <>
              <h3>Style</h3>
              {memory.style_rules.map((rule: any) => (
                <div className="card" key={String(rule.key)} style={{ padding: 12 }}>
                  <div className="card-row">
                    <div>
                      <strong>{String(rule.value)}</strong>
                      <div className="small dim">
                        seen in {rule.evidence_count} approvals · {String(rule.promoted_by)}
                      </div>
                      {Array.isArray(rule.evidence) && rule.evidence.length > 0 && (
                        <div className="small dim mono">
                          {rule.evidence.slice(0, 5).join(", ")}
                        </div>
                      )}
                    </div>
                    <div className="btn-row">
                      <button
                        style={{ minHeight: 32, padding: "4px 10px" }}
                        disabled={busy === rule.key}
                        onClick={() => forget("style", String(rule.key), false)}
                      >
                        Stop applying
                      </button>
                      <button
                        className="btn-danger"
                        style={{ minHeight: 32, padding: "4px 10px" }}
                        disabled={busy === rule.key}
                        onClick={() => forget("style", String(rule.key), true)}
                      >
                        Forget entirely
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </>
          )}

          {memory.outcome_rules.length > 0 && (
            <>
              <h3>Outcomes</h3>
              {memory.outcome_rules.map((rule: any) => (
                <div className="card" key={String(rule.key)} style={{ padding: 12 }}>
                  <div className="card-row">
                    <div>
                      <strong>{String(rule.key)}</strong>
                      <div className="small">{String(rule.value?.note ?? "")}</div>
                      <div className="small dim">{String(rule.promoted_by)}</div>
                    </div>
                    <button
                      className="btn-danger"
                      style={{ minHeight: 32, padding: "4px 10px" }}
                      disabled={busy === rule.key}
                      onClick={() => forget("outcome", String(rule.key), true)}
                    >
                      Forget
                    </button>
                  </div>
                </div>
              ))}
            </>
          )}
        </>
      )}
    </>
  );
}
