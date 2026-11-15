"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { api, type PipelineBoard } from "@/lib/api/client";
import { Empty, ErrorState, Loading } from "@/components/States";

const ORDER = ["drafted", "submitted", "viewed", "replied", "interview", "hired", "lost"];

const HINTS: Record<string, string> = {
  drafted: "Waiting for you. Nothing spent yet.",
  submitted: "Sent, no response yet.",
  viewed: "The client opened it.",
  replied: "They wrote back.",
  interview: "A call is on the table.",
  hired: "Won.",
  lost: "Declined or closed.",
};

/**
 * The pipeline board.
 *
 * Cards move because the marketplace said so, not because we decided they had. The only
 * column we own is "drafted": a proposal the user has not approved, which the marketplace
 * has never heard of and which is the most actionable card on the board.
 */
export default function PipelinePage() {
  const [board, setBoard] = useState<PipelineBoard | null>(null);
  const [error, setError] = useState<unknown>(null);

  const load = useCallback(() => {
    setError(null);
    api.pipeline().then(setBoard).catch(setError);
  }, []);

  useEffect(load, [load]);

  if (error) return <ErrorState error={error} retry={load} />;
  if (!board) return <Loading what="Loading the pipeline" />;

  const total = Object.values(board.totals).reduce((sum, n) => sum + n, 0);

  return (
    <>
      <h1>Pipeline</h1>
      <p className="subtitle">
        {total === 0
          ? "Nothing in flight yet."
          : `${total} ${total === 1 ? "proposal" : "proposals"} across the funnel.`}
      </p>

      {total === 0 ? (
        <Empty
          title="No proposals yet"
          hint="Analyse a job from the inbox, then approve the draft to put it on the board."
        />
      ) : (
        <div className="kanban">
          {ORDER.map((column) => {
            const cards = board.columns[column] ?? [];
            return (
              <div className="kanban-col" key={column}>
                <h3>
                  {column} ({cards.length})
                </h3>
                {cards.length === 0 ? (
                  <p className="small dim" style={{ margin: 0 }}>
                    {HINTS[column]}
                  </p>
                ) : (
                  cards.map((card: any, index: number) => (
                    <div
                      className="card"
                      key={`${card.job_id}-${index}`}
                      style={{ padding: 10, marginBottom: 8 }}
                    >
                      <div className="small" style={{ fontWeight: 600 }}>
                        {card.title ?? card.job_id}
                      </div>
                      <div className="small dim mono">{card.job_id}</div>
                      <div className="small dim">
                        {card.price_usd ? `$${Number(card.price_usd).toFixed(0)}` : ""}
                        {card.connects_spent ? ` · ${card.connects_spent} Connects` : ""}
                        {card.fit_coverage !== null && card.fit_coverage !== undefined
                          ? ` · ${Math.round(card.fit_coverage * 100)}% covered`
                          : ""}
                      </div>
                      {card.thread_id && column === "drafted" && (
                        <Link
                          className="small"
                          href={`/studio/${encodeURIComponent(card.thread_id)}`}
                        >
                          Review →
                        </Link>
                      )}
                      {card.job_id && column !== "drafted" && (
                        <Link
                          className="small"
                          href={`/jobs/${encodeURIComponent(card.job_id)}`}
                        >
                          Open job →
                        </Link>
                      )}
                    </div>
                  ))
                )}
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}
