"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type Analytics } from "@/lib/api/client";
import { Empty, ErrorState, Loading } from "@/components/States";

/**
 * Analytics: reply and hire rate, Connects against outcomes, cost and latency per
 * proposal, and the top reasons jobs were skipped.
 *
 * Reply and hire rate come from the marketplace's own records, not from our threads.
 * Taking them from our state would mean reporting our opinion of the outcome rather than
 * what the client actually did. Skip reasons are the opposite: the marketplace never
 * hears about a job we chose not to bid on, so those can only come from our own
 * decisions.
 *
 * A rate over a handful of proposals is noise, so the denominator is always shown.
 */
export default function AnalyticsPage() {
  const [data, setData] = useState<Analytics | null>(null);
  const [error, setError] = useState<unknown>(null);

  const load = useCallback(() => {
    setError(null);
    api.analytics().then(setData).catch(setError);
  }, []);

  useEffect(load, [load]);

  if (error) return <ErrorState error={error} retry={load} />;
  if (!data) return <Loading what="Loading analytics" />;

  const pct = (value?: number | null) =>
    value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;

  return (
    <>
      <h1>Analytics</h1>
      <p className="subtitle">
        Outcomes come from the marketplace. Skip reasons come from our own decisions,
        because the marketplace never sees a job we did not bid on.
      </p>

      {data.proposals_submitted === 0 ? (
        <Empty
          title="Nothing submitted yet"
          hint="Approve a proposal and the funnel will start filling in."
        />
      ) : (
        <>
          <div className="grid grid-3">
            <div className="stat">
              <div className="stat-value">{data.proposals_submitted}</div>
              <div className="stat-label">Submitted</div>
            </div>
            <div className="stat">
              <div className="stat-value">{pct(data.reply_rate)}</div>
              <div className="stat-label">
                Reply rate · {data.replies}/{data.proposals_submitted}
              </div>
            </div>
            <div className="stat">
              <div className="stat-value">{pct(data.hire_rate)}</div>
              <div className="stat-label">
                Hire rate · {data.hires}/{data.proposals_submitted}
              </div>
            </div>
            <div className="stat">
              <div className="stat-value">{data.connects_spent}</div>
              <div className="stat-label">
                Connects spent
                {data.connects_remaining !== null && data.connects_remaining !== undefined
                  ? ` · ${data.connects_remaining} left this week`
                  : ""}
              </div>
            </div>
          </div>

          {data.proposals_submitted < 10 && (
            <p className="small dim" style={{ marginTop: 10 }}>
              A rate over {data.proposals_submitted} proposals is not a trend. Treat these
              as a tally, not a measurement.
            </p>
          )}
        </>
      )}

      <h2>Cost and speed</h2>
      <div className="grid grid-3">
        <div className="stat">
          <div className="stat-value">
            {data.cost.mean_tokens_per_run
              ? Math.round(Number(data.cost.mean_tokens_per_run)).toLocaleString()
              : "—"}
          </div>
          <div className="stat-label">Tokens per run · {data.cost.runs ?? 0} runs</div>
        </div>
        <div className="stat">
          <div className="stat-value">
            {data.latency.node_p50_ms ? `${Math.round(Number(data.latency.node_p50_ms))}ms` : "—"}
          </div>
          <div className="stat-label">Node p50</div>
        </div>
        <div className="stat">
          <div className="stat-value">
            {data.latency.node_p95_ms ? `${Math.round(Number(data.latency.node_p95_ms))}ms` : "—"}
          </div>
          <div className="stat-label">Node p95 · {data.latency.samples ?? 0} samples</div>
        </div>
        <div className="stat">
          <div className="stat-value">{pct(data.unsupported_claim_rate)}</div>
          <div className="stat-label">Drafts with an unsupported claim</div>
        </div>
      </div>

      {data.top_skip_reasons.length > 0 && (
        <>
          <h2>Why jobs were skipped</h2>
          <div className="card" style={{ padding: 0 }}>
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Deciding factor</th>
                    <th scope="col" className="num">
                      Jobs
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.top_skip_reasons.map((row: any) => (
                    <tr key={String(row.reason)}>
                      <td>{String(row.reason).replace(/_/g, " ")}</td>
                      <td className="num">{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {data.by_category.length > 0 && (
        <>
          <h2>By category</h2>
          <div className="card" style={{ padding: 0 }}>
            <div className="scroll-x">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Category</th>
                    <th scope="col" className="num">
                      Submitted
                    </th>
                    <th scope="col" className="num">
                      Replied
                    </th>
                    <th scope="col" className="num">
                      Hired
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.by_category.map((row: any) => (
                    <tr key={String(row.category)}>
                      <td>{String(row.category)}</td>
                      <td className="num">{row.submitted}</td>
                      <td className="num">{row.replied}</td>
                      <td className="num">{row.hired}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  );
}
