"use client";

/**
 * Node-by-node progress, live or from a finished run's trace.
 *
 * Shows the prompt version next to each model node (`writer@v3`), because that is what
 * makes "the Writer got worse last Tuesday" answerable, and the token count, because
 * cost per proposal is a declared budget.
 */

type Entry = {
  node: string;
  kind?: string | null;
  tool?: string | null;
  prompt?: string | null;
  tokens?: number | null;
  duration_ms?: number | null;
  summary?: unknown;
  result_summary?: unknown;
  error?: string | null;
};

const KIND_LABEL: Record<string, string> = {
  tool_call: "tool",
  model: "model",
  decision: "decision",
  guardrail: "guardrail",
  approval: "approval",
  error: "error",
};

export function RunTimeline({ entries, live }: { entries: Entry[]; live?: boolean }) {
  if (entries.length === 0) {
    return <p className="dim small">No steps yet.</p>;
  }

  return (
    <ol className="timeline">
      {entries.map((entry, index) => {
        const summary = entry.summary ?? entry.result_summary;
        return (
          <li key={index}>
            <div>
              <span className="node">{entry.node}</span>{" "}
              <span className="pill">{KIND_LABEL[entry.kind ?? ""] ?? entry.kind}</span>
            </div>
            <div className="small dim" style={{ minWidth: 0 }}>
              {entry.error ? (
                <span style={{ color: "var(--danger)" }}>{entry.error}</span>
              ) : (
                <span className="mono">
                  {summary ? JSON.stringify(summary).slice(0, 220) : entry.tool ?? ""}
                </span>
              )}
              {entry.prompt && <span className="pill" style={{ marginLeft: 6 }}>{entry.prompt}</span>}
            </div>
            <div className="small dim num" style={{ whiteSpace: "nowrap" }}>
              {entry.tokens ? `${entry.tokens} tok` : ""}
              {entry.duration_ms ? ` · ${Math.round(entry.duration_ms)}ms` : ""}
            </div>
          </li>
        );
      })}
      {live && (
        <li>
          <div>
            <span className="spinner" aria-hidden="true" />
          </div>
          <div className="small dim">running…</div>
          <div />
        </li>
      )}
    </ol>
  );
}
