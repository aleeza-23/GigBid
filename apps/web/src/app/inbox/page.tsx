"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, type JobPage, type PendingApproval } from "@/lib/api/client";
import { ActionBadge, CoverageBadge, RiskBadge, StatusBadge } from "@/components/Badges";
import { Empty, ErrorState, Loading } from "@/components/States";

/**
 * The job inbox.
 *
 * Two tiers, shown differently on purpose. A job that has been analysed has real coverage
 * and risk; one that has not is ranked by skill overlap and says "not analysed" rather
 * than implying a score of zero. Running the full graph on every job in the feed would
 * cost tokens on jobs the user will never open.
 *
 * Anything waiting for a decision is pulled to the top, because that is the only thing on
 * this screen blocked on the human.
 */
export default function InboxPage() {
  const [page, setPage] = useState<JobPage | null>(null);
  const [pending, setPending] = useState<PendingApproval[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [query, setQuery] = useState("");

  const load = useCallback(() => {
    setError(null);
    api.jobs({ limit: 30, q: query || undefined }).then(setPage).catch(setError);
    api
      .approvals()
      .then(setPending)
      .catch(() => setPending([]));
  }, [query]);

  useEffect(load, [load]);

  if (error) return <ErrorState error={error} retry={load} />;

  return (
    <>
      <h1>Job inbox</h1>
      <p className="subtitle">
        Ranked by requirement coverage where we have analysed the job, by skill overlap
        where we have not.
      </p>

      {pending.length > 0 && (
        <div className="banner banner-warn" role="status">
          <strong>
            {pending.length} {pending.length === 1 ? "item" : "items"} waiting for you.
          </strong>{" "}
          {pending.map((item) => (
            <Link
              key={item.thread_id}
              href={
                item.kind === "message"
                  ? "/messages"
                  : `/studio/${encodeURIComponent(item.thread_id)}`
              }
              style={{ marginRight: 10 }}
            >
              {item.title ?? item.job_id ?? item.thread_id}
            </Link>
          ))}
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          load();
        }}
        style={{ display: "flex", gap: 8, marginBottom: 14, maxWidth: 480 }}
      >
        <label htmlFor="q" style={{ position: "absolute", left: -9999 }}>
          Search jobs
        </label>
        <input
          id="q"
          placeholder="Search jobs…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit">Search</button>
      </form>

      {!page ? (
        <Loading what="Loading jobs" />
      ) : page.jobs.length === 0 ? (
        <Empty
          title="No jobs match"
          hint="Try a broader search, or check that GigSim is running."
        />
      ) : (
        <div className="scroll-x card" style={{ padding: 0 }}>
          <table>
            <caption
              className="small dim"
              style={{ padding: "10px 10px 0", textAlign: "left" }}
            >
              {page.jobs.length} of {page.total} jobs · {page.marketplace}
            </caption>
            <thead>
              <tr>
                <th scope="col">Job</th>
                <th scope="col">Fit</th>
                <th scope="col">Risk</th>
                <th scope="col">Action</th>
                <th scope="col" className="num">
                  Price
                </th>
                <th scope="col" className="num">
                  Connects
                </th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {page.jobs.map((job) => (
                <tr key={job.job_id}>
                  <td>
                    <Link href={`/jobs/${encodeURIComponent(job.job_id)}`}>{job.title}</Link>
                    <div className="small dim">
                      {job.skills.slice(0, 4).join(" · ")}
                      {job.fit_coverage === null && job.skill_overlap > 0 && (
                        <> · {job.skill_overlap} skills match</>
                      )}
                    </div>
                  </td>
                  <td>
                    <CoverageBadge coverage={job.fit_coverage} />
                  </td>
                  <td>
                    <RiskBadge score={job.risk_score} />
                  </td>
                  <td>
                    <ActionBadge action={job.recommended_action} />
                  </td>
                  <td className="num">
                    {job.recommended_price ? `$${job.recommended_price.toFixed(0)}` : "—"}
                  </td>
                  <td className="num">{job.connects_cost ?? "—"}</td>
                  <td>
                    <StatusBadge status={job.thread_status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
