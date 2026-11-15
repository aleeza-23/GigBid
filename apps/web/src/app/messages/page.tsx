"use client";

import { useCallback, useEffect, useState } from "react";
import {
  api,
  streamRun,
  type MessageThread,
  type ThreadSummary,
} from "@/lib/api/client";
import { Empty, ErrorState, Loading } from "@/components/States";

/**
 * Client threads, with suggested replies and interview slots.
 *
 * Drafting a reply runs the conversation graph, which stops at an interrupt exactly as
 * the bid graph does. Two outcomes the screen has to distinguish:
 *
 * - a draft waiting for approval, which is the normal case;
 * - a **scam signal**, where no reply was drafted at all. The graph refuses rather than
 *   writing a polite answer, because a polite answer to a scam is one a hurried user
 *   might approve.
 */
export default function MessagesPage() {
  const [threads, setThreads] = useState<ThreadSummary[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [thread, setThread] = useState<MessageThread | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [drafting, setDrafting] = useState(false);
  const [edited, setEdited] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<string | null>(null);

  const loadThreads = useCallback(() => {
    setError(null);
    api.threads().then(setThreads).catch(setError);
  }, []);

  useEffect(loadThreads, [loadThreads]);

  const openThread = useCallback((id: string) => {
    setSelected(id);
    setThread(null);
    setOutcome(null);
    setEdited(null);
    api.thread(id).then(setThread).catch(setError);
  }, []);

  async function draft() {
    if (!selected) return;
    setDrafting(true);
    setOutcome(null);
    try {
      await streamRun(
        `/messages/${encodeURIComponent(selected)}/draft`,
        () => {
          // Progress is shown as a spinner here; the run viewer is the detailed feed.
        },
        { method: "POST" },
      );
      const refreshed = await api.thread(selected);
      setThread(refreshed);
      setEdited(refreshed.draft_reply ?? null);
    } catch (err) {
      setError(err);
    } finally {
      setDrafting(false);
    }
  }

  async function decide(action: "approve" | "edit" | "reject") {
    if (!selected) return;
    // The approval thread id is namespaced; the API scopes it by the session's user.
    const approvals = await api.approvals().catch(() => []);
    const pending = approvals.find((item) => item.thread_id.endsWith(`:${selected}`));
    if (!pending) {
      setOutcome("This thread is not waiting for approval.");
      return;
    }
    setBusy(action);
    try {
      const result = await api.decide(pending.thread_id, {
        action,
        text: action === "edit" ? (edited ?? "") : undefined,
      });
      setOutcome(
        result.status === "submitted"
          ? `Sent.${result.client_reply ? " The client replied already." : ""}`
          : `Ended as ${result.status}. Nothing was sent.`,
      );
      openThread(selected);
      loadThreads();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(null);
    }
  }

  if (error && !threads) return <ErrorState error={error} retry={loadThreads} />;

  return (
    <>
      <h1>Messages</h1>
      <p className="subtitle">
        Replies are drafted for you to approve. Nothing is sent on its own.
      </p>

      {error && <ErrorState error={error} />}

      {!threads ? (
        <Loading what="Loading threads" />
      ) : threads.length === 0 ? (
        <Empty
          title="No client conversations yet"
          hint="Threads appear once a client replies to a submitted proposal."
        />
      ) : (
        <div className="grid" style={{ gridTemplateColumns: "minmax(240px, 320px) 1fr" }}>
          <div>
            {threads.map((item) => (
              <button
                key={item.thread_id}
                onClick={() => openThread(item.thread_id)}
                aria-current={selected === item.thread_id ? "true" : undefined}
                style={{
                  width: "100%",
                  textAlign: "left",
                  marginBottom: 8,
                  background:
                    selected === item.thread_id ? "var(--surface-2)" : "var(--surface)",
                }}
              >
                <span className="small" style={{ fontWeight: 600 }}>
                  {item.job_id ?? item.thread_id}
                </span>
                {item.awaiting_approval && (
                  <span className="pill pill-warn" style={{ marginLeft: 6 }}>
                    awaiting you
                  </span>
                )}
                <span className="small dim" style={{ display: "block" }}>
                  {item.last_from}: {(item.last_text ?? "").slice(0, 70)}
                </span>
              </button>
            ))}
          </div>

          <div>
            {!selected ? (
              <Empty title="Pick a conversation" />
            ) : !thread ? (
              <Loading what="Loading conversation" />
            ) : (
              <>
                {outcome && (
                  <div className="banner" role="status" style={{ background: "var(--surface-2)" }}>
                    {outcome}
                  </div>
                )}

                {thread.risk_flags.length > 0 && (
                  <div className="banner banner-danger" role="alert">
                    <strong>Policy signals in this conversation:</strong>{" "}
                    {thread.risk_flags.join("; ")}
                  </div>
                )}

                {thread.intent === "scam_signal" && (
                  <div className="banner banner-danger" role="alert">
                    <strong>No reply was drafted.</strong> The client&apos;s message asks
                    for something the policy pack forbids. Answering it politely is worse
                    than not answering.
                  </div>
                )}

                <section className="card">
                  <h3 style={{ marginTop: 0 }}>Conversation</h3>
                  {thread.messages.map((message: any, index: number) => (
                    <div
                      key={index}
                      style={{
                        padding: "8px 10px",
                        marginBottom: 6,
                        borderRadius: 8,
                        background:
                          (message.from ?? message.sender) === "client"
                            ? "var(--surface-2)"
                            : "transparent",
                        border: "1px solid var(--border)",
                      }}
                    >
                      <div className="small dim">{message.from ?? message.sender}</div>
                      <div style={{ fontSize: 14 }}>{message.text}</div>
                    </div>
                  ))}
                </section>

                <section className="card" style={{ marginTop: 12 }}>
                  <div className="card-row">
                    <h3 style={{ margin: 0 }}>Suggested reply</h3>
                    {thread.intent && <span className="pill">{thread.intent}</span>}
                  </div>

                  {thread.proposed_slots.length > 0 && (
                    <p className="small">
                      Offering:{" "}
                      {thread.proposed_slots.map((slot: any) => (
                        <span key={slot.start} className="pill" style={{ marginRight: 4 }}>
                          {new Date(slot.start).toLocaleString()}
                        </span>
                      ))}
                    </p>
                  )}

                  {thread.draft_reply ? (
                    <>
                      <label htmlFor="reply">Reply (edit before sending if you like)</label>
                      <textarea
                        id="reply"
                        style={{ minHeight: 140 }}
                        value={edited ?? thread.draft_reply}
                        onChange={(e) => setEdited(e.target.value)}
                      />
                      <div className="btn-row">
                        <button
                          className="btn-primary"
                          disabled={busy !== null}
                          onClick={() =>
                            decide(
                              edited !== null && edited !== thread.draft_reply
                                ? "edit"
                                : "approve",
                            )
                          }
                        >
                          {busy ? "Sending…" : "Approve and send"}
                        </button>
                        <button
                          className="btn-danger"
                          disabled={busy !== null}
                          onClick={() => decide("reject")}
                        >
                          Discard
                        </button>
                      </div>
                    </>
                  ) : (
                    <>
                      <p className="small dim">No reply drafted yet.</p>
                      <button className="btn-primary" onClick={draft} disabled={drafting}>
                        {drafting ? (
                          <>
                            <span className="spinner" aria-hidden="true" /> Drafting…
                          </>
                        ) : (
                          "Draft a reply"
                        )}
                      </button>
                    </>
                  )}
                </section>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}
