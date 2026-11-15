"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, streamRun, type RunDetail, type TraceEntry } from "@/lib/api/client";
import { RunTimeline } from "@/components/RunTimeline";
import { ErrorState, Loading } from "@/components/States";

/**
 * The agent run viewer: node-by-node progress, live.
 *
 * Opening this page attaches to the run's SSE stream, which either starts the graph (a
 * queued run) or replays its current position (an already-finished or already-waiting
 * one). Either way the panel fills from the thread's stored trace first, so a refresh
 * mid-run shows history rather than an empty box — the live feed is an addition to the
 * record, never the only copy of it.
 *
 * When the graph interrupts, the stream ends with `awaiting_approval` and this page sends
 * the user to the proposal studio rather than leaving them on a stalled progress bar.
 */
export default function RunPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const threadId = decodeURIComponent(params.id);

  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [live, setLive] = useState<TraceEntry[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [finished, setFinished] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<unknown>(null);
  const started = useRef(false);

  const loadDetail = useCallback(() => {
    api
      .run(threadId)
      .then(setDetail)
      .catch((err) => {
        // A queued run has no state yet; that is not an error worth showing.
        if (!String(err?.message ?? "").includes("no such run")) setError(err);
      });
  }, [threadId]);

  useEffect(loadDetail, [loadDetail]);

  useEffect(() => {
    if (started.current) return;
    started.current = true;

    const controller = new AbortController();
    setStreaming(true);

    streamRun(
      `/runs/${encodeURIComponent(threadId)}/stream`,
      (event) => {
        if (event.event === "node" || event.event === "node_error") {
          setLive((entries) => [...entries, event.data as TraceEntry]);
        } else if (event.event === "awaiting_approval") {
          setStreaming(false);
          router.push(`/studio/${encodeURIComponent(threadId)}`);
        } else if (event.event === "run_finished") {
          setStreaming(false);
          setFinished(event.data);
          loadDetail();
        } else if (event.event === "stream_error") {
          setError(new Error(String((event.data as any).error)));
          setStreaming(false);
        }
      },
      { signal: controller.signal },
    )
      .catch((err) => {
        if (!controller.signal.aborted) setError(err);
      })
      .finally(() => setStreaming(false));

    return () => controller.abort();
  }, [threadId, router, loadDetail]);

  // The stored trace is the record; live events are what has arrived since it was read.
  const stored = detail?.trace ?? [];
  const entries = live.length > stored.length ? live : stored;

  return (
    <>
      <p className="small">
        <Link href="/inbox">← Inbox</Link>
        {detail?.summary.job_id && (
          <>
            {" · "}
            <Link href={`/jobs/${encodeURIComponent(detail.summary.job_id)}`}>
              {detail.summary.job_id}
            </Link>
          </>
        )}
      </p>
      <h1>Agent run</h1>
      <p className="subtitle">
        <span className="mono">{threadId}</span>
        {detail?.summary.trace_id && (
          <>
            {" · trace "}
            <span className="mono">{detail.summary.trace_id}</span>
          </>
        )}
      </p>

      {detail?.summary.awaiting_approval && (
        <p>
          <Link
            className="btn btn-primary"
            href={`/studio/${encodeURIComponent(threadId)}`}
          >
            Review the proposal
          </Link>
        </p>
      )}

      {detail?.summary.status === "needs_attention" && detail.summary.job_id && (
        <p>
          <button
            className="btn"
            onClick={async () => {
              try {
                const retry = await api.startRun(detail.summary.job_id!, true);
                router.push(`/runs/${encodeURIComponent(retry.thread_id)}`);
              } catch (err) {
                setError(err);
              }
            }}
          >
            Retry this run
          </button>
        </p>
      )}

      {error && <ErrorState error={error} />}

      {detail?.summary.budget_exceeded && (
        <div className="banner banner-warn" role="status">
          <strong>Stopped on budget.</strong> {detail.summary.budget_exceeded}. The thread
          is marked for attention rather than left looping.
        </div>
      )}

      {detail && Object.keys(detail.summary.degraded ?? {}).length > 0 && (
        <div className="banner banner-warn" role="status">
          <strong>Ran degraded.</strong>{" "}
          {Object.entries(detail.summary.degraded)
            .map(([key, value]) => `${key}: ${String(value).slice(0, 120)}`)
            .join(" · ")}
        </div>
      )}

      {finished && (
        <div className="banner" style={{ background: "var(--surface-2)" }} role="status">
          <strong>Finished: {String(finished.status)}</strong>
          {finished.decision ? ` · decided to ${String(finished.decision)}` : ""}
          {Array.isArray(finished.reasons) && finished.reasons.length > 0 && (
            <ul className="small" style={{ paddingLeft: 18, margin: "6px 0 0" }}>
              {(finished.reasons as string[]).filter(Boolean).map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {detail && (
        <div className="grid grid-3" style={{ marginBottom: 14 }}>
          <div className="stat">
            <div className="stat-value">{detail.summary.steps_used}</div>
            <div className="stat-label">Steps</div>
          </div>
          <div className="stat">
            <div className="stat-value">{detail.summary.tokens_used.toLocaleString()}</div>
            <div className="stat-label">Tokens</div>
          </div>
          <div className="stat">
            <div className="stat-value">{detail.summary.revision_count}</div>
            <div className="stat-label">Critic revisions</div>
          </div>
          <div className="stat">
            <div className="stat-value">
              {detail.summary.unsupported_claims ?? "—"}
            </div>
            <div className="stat-label">Unsupported claims</div>
          </div>
        </div>
      )}

      <h2>Steps</h2>
      {entries.length === 0 && streaming ? (
        <Loading what="Starting the graph" />
      ) : (
        <div className="card">
          <RunTimeline entries={entries} live={streaming} />
        </div>
      )}

      {detail && detail.errors.length > 0 && (
        <>
          <h2>Errors</h2>
          <div className="card">
            <ul className="small" style={{ paddingLeft: 18, margin: 0 }}>
              {detail.errors.map((err, index) => (
                <li key={index}>
                  <strong>{String(err.node)}</strong>: {String(err.error)}
                  {err.fatal ? " (fatal)" : ""}
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </>
  );
}
