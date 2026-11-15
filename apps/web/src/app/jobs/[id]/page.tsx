"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { api, type JobDetail } from "@/lib/api/client";
import { CoverageBadge, RiskBadge } from "@/components/Badges";
import { ErrorState, Loading } from "@/components/States";

/**
 * Job detail: the spec, the risk report, the requirement-to-evidence matrix, the price
 * rationale and the bid-or-skip reasoning.
 *
 * The matrix is the screen's reason for existing. A coverage percentage is a number to
 * argue with; a list of requirements each with the portfolio items that prove it is
 * something a freelancer can check. An uncovered requirement is the useful half: it is
 * what the proposal must not claim.
 */
export default function JobDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const jobId = decodeURIComponent(params.id);

  const [detail, setDetail] = useState<JobDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [starting, setStarting] = useState(false);

  const load = useCallback(() => {
    setError(null);
    api.job(jobId).then(setDetail).catch(setError);
  }, [jobId]);

  useEffect(load, [load]);

  async function analyse() {
    setStarting(true);
    try {
      const run = await api.startRun(jobId);
      router.push(`/runs/${encodeURIComponent(run.thread_id)}`);
    } catch (err) {
      setError(err);
      setStarting(false);
    }
  }

  if (error) return <ErrorState error={error} retry={load} />;
  if (!detail) return <Loading what="Loading job" />;

  const job = detail.job;
  const decision = detail.decision;

  return (
    <>
      <p className="small">
        <Link href="/inbox">← Inbox</Link>
      </p>
      <h1>{job.title}</h1>
      <p className="subtitle">
        <span className="mono">{jobId}</span>
        {job.category ? ` · ${job.category}` : ""}
        {job.country ? ` · ${job.country}` : ""}
      </p>

      {detail.injection_removed && (
        <div className="banner banner-warn" role="status">
          <strong>Instruction-like text was removed from this post.</strong> Something in
          it was addressed to an automated reader rather than to a freelancer. It was
          stripped before any agent read the text, and treated as a risk signal.
        </div>
      )}

      <div className="btn-row" style={{ marginBottom: 16 }}>
        <button className="btn-primary" onClick={analyse} disabled={starting}>
          {starting ? "Starting…" : detail.analysed ? "Re-run analysis" : "Analyse this job"}
        </button>
        {detail.thread_id && (
          <Link className="btn" href={`/runs/${encodeURIComponent(detail.thread_id)}`}>
            View run
          </Link>
        )}
        {detail.thread_status === "awaiting_approval" && detail.thread_id && (
          <Link className="btn btn-primary" href={`/studio/${encodeURIComponent(detail.thread_id)}`}>
            Review the proposal
          </Link>
        )}
      </div>

      {!detail.analysed && (
        <div className="banner" style={{ background: "var(--surface-2)" }}>
          This job has not been analysed yet. Running the analysis costs a few thousand
          tokens and no Connects — nothing is sent without your approval.
        </div>
      )}

      <div className="grid grid-2">
        <section className="card">
          <h3 style={{ marginTop: 0 }}>The post</h3>
          <p className="small dim">
            {job.is_hourly
              ? `Hourly $${job.hourly_low ?? "?"}–$${job.hourly_high ?? "?"}`
              : `Fixed $${job.budget ?? "?"}`}
            {job.connects_cost ? ` · ${job.connects_cost} Connects to apply` : ""}
          </p>
          <p style={{ whiteSpace: "pre-wrap", fontSize: 14 }}>
            {String(job.description ?? "").slice(0, 1400)}
            {String(job.description ?? "").length > 1400 ? "…" : ""}
          </p>
          {Array.isArray(job.skills) && job.skills.length > 0 && (
            <p className="small">
              {job.skills.map((skill: string) => (
                <span key={skill} className="pill" style={{ marginRight: 4 }}>
                  {skill}
                </span>
              ))}
            </p>
          )}
        </section>

        <section className="card">
          <h3 style={{ marginTop: 0 }}>The client</h3>
          {detail.client?.client_id ? (
            <>
              <p className="small">
                <RiskBadge score={detail.risk?.score} />
              </p>
              <table>
                <tbody>
                  <tr>
                    <td>Jobs posted</td>
                    <td className="num">{detail.client.jobs_posted ?? "—"}</td>
                  </tr>
                  <tr>
                    <td>Hires</td>
                    <td className="num">{detail.client.hires ?? "—"}</td>
                  </tr>
                  <tr>
                    <td>Hire rate</td>
                    <td className="num">
                      {detail.client.hire_rate !== null && detail.client.hire_rate !== undefined
                        ? `${Math.round(detail.client.hire_rate * 100)}%`
                        : "—"}
                    </td>
                  </tr>
                  <tr>
                    <td>Total spent</td>
                    <td className="num">
                      {detail.client.total_spent_usd
                        ? `$${Number(detail.client.total_spent_usd).toFixed(0)}`
                        : "—"}
                    </td>
                  </tr>
                  <tr>
                    <td>Payment verified</td>
                    <td className="num">
                      {detail.client.payment_verified === true
                        ? "yes"
                        : detail.client.payment_verified === false
                          ? "no"
                          : "—"}
                    </td>
                  </tr>
                </tbody>
              </table>
              {(detail.risk?.flags ?? []).length > 0 && (
                <ul className="small" style={{ paddingLeft: 18, marginBottom: 0 }}>
                  {detail.risk!.flags.map((flag: string) => (
                    <li key={flag}>{flag}</li>
                  ))}
                </ul>
              )}
            </>
          ) : (
            <p className="dim small">Client history unavailable.</p>
          )}
        </section>
      </div>

      {decision && (
        <section className="card" style={{ marginTop: 12 }}>
          <div className="card-row">
            <div>
              <h3 style={{ margin: 0 }}>
                Recommendation: {decision.decision}
                {decision.deciding_factor ? (
                  <span className="pill" style={{ marginLeft: 8 }}>
                    decided on {decision.deciding_factor}
                  </span>
                ) : null}
              </h3>
              <ul className="small" style={{ paddingLeft: 18, marginBottom: 0 }}>
                {(decision.reasons ?? []).filter(Boolean).map((reason: string) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
            <CoverageBadge coverage={detail.fit?.coverage} />
          </div>
        </section>
      )}

      {detail.price && (
        <section className="card" style={{ marginTop: 12 }}>
          <h3 style={{ marginTop: 0 }}>
            Price: ${Number(detail.price.amount_usd).toFixed(0)} {detail.price.type}
          </h3>
          <p className="small">{detail.price.rationale}</p>
          <p className="small dim">
            Floor ${detail.price.profile_floor_usd ?? "—"} · market p50 $
            {detail.price.market_p50_usd ?? "—"}
            {detail.price.market_sample_size
              ? ` from ${detail.price.market_sample_size} contracts`
              : ""}
            {detail.price.above_floor === false && (
              <strong style={{ color: "var(--danger)" }}> · below the floor</strong>
            )}
          </p>
        </section>
      )}

      {detail.requirement_matrix.length > 0 && (
        <section style={{ marginTop: 12 }}>
          <h2>Requirement to evidence</h2>
          <p className="subtitle">
            What the client asked for, and the portfolio items that prove each one. An
            uncovered requirement is one the proposal must not claim.
          </p>
          <div className="card" style={{ padding: 0 }}>
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Requirement</th>
                    <th scope="col">Covered</th>
                    <th scope="col">Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.requirement_matrix.map((row) => (
                    <tr key={row.requirement}>
                      <td>{row.requirement}</td>
                      <td>
                        <span className={`pill ${row.covered ? "pill-ok" : "pill-danger"}`}>
                          {row.covered ? "covered" : "no evidence"}
                        </span>
                        {row.overruled_by_model && (
                          <div className="small dim">{row.overruled_by_model}</div>
                        )}
                      </td>
                      <td className="small">
                        {row.evidence.length === 0 ? (
                          <span className="dim">nothing retrieved</span>
                        ) : (
                          row.evidence.slice(0, 3).map((item: any) => (
                            <div key={String(item.id)}>
                              <span className="mono">{item.id}</span>{" "}
                              <span className="dim">
                                {item.project ? `${item.project} · ` : ""}
                                {item.type}
                              </span>
                            </div>
                          ))
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}
    </>
  );
}
