"use client";

/**
 * A proposal rendered claim by claim, with the evidence behind each on hover.
 *
 * Part D requires "every claim shows its evidence on hover". Two details make that
 * actually usable:
 *
 * - it works from the keyboard. `tabIndex` plus `:focus-within` in the CSS means tabbing
 *   through claims reveals each popover, so this is not a mouse-only feature;
 * - a claim whose citation did not resolve is marked. That is the R5 failure mode made
 *   visible: a sentence citing evidence that was never retrieved is exactly what the
 *   verifier exists to catch, and the user should see when one slipped through.
 */

type Claim = {
  claim: string;
  evidence: Array<{
    id?: string;
    citation?: string | null;
    project?: string | null;
    type?: string | null;
    text?: string | null;
    resolved?: boolean;
  }>;
};

export function EvidenceHover({
  text,
  claims,
}: {
  text: string;
  claims: Claim[];
}) {
  // Longest first, so a claim that contains another is matched before its substring.
  const sorted = [...claims].sort((a, b) => b.claim.length - a.claim.length);

  // Split the proposal into cited and uncited runs, preserving order.
  const segments: Array<{ text: string; claim?: Claim }> = [];
  let rest = text;

  const consume = (chunk: string) => {
    for (const claim of sorted) {
      const at = chunk.indexOf(claim.claim);
      if (at === -1) continue;
      if (at > 0) segments.push({ text: chunk.slice(0, at) });
      segments.push({ text: claim.claim, claim });
      return chunk.slice(at + claim.claim.length);
    }
    segments.push({ text: chunk });
    return "";
  };

  let guard = 0;
  while (rest && guard++ < 200) rest = consume(rest);

  return (
    <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}>
      {segments.map((segment, index) =>
        segment.claim ? (
          <span key={index} className="claim" tabIndex={0}>
            {segment.text}
            <span className="claim-pop" role="note">
              <strong className="small">Evidence for this claim</strong>
              {segment.claim.evidence.length === 0 && (
                <p className="small" style={{ color: "var(--danger)", margin: "6px 0 0" }}>
                  No evidence attached. This claim is unsupported.
                </p>
              )}
              {segment.claim.evidence.map((item, j) => (
                <div key={j} style={{ marginTop: 8 }}>
                  <div className="mono small">
                    {item.citation ?? item.id}
                    {item.resolved === false && (
                      <span className="pill pill-danger" style={{ marginLeft: 6 }}>
                        not retrieved
                      </span>
                    )}
                  </div>
                  {item.project && <div className="small dim">{item.project} · {item.type}</div>}
                  {item.text && (
                    <p className="small" style={{ margin: "4px 0 0" }}>
                      {item.text.slice(0, 280)}
                      {item.text.length > 280 ? "…" : ""}
                    </p>
                  )}
                </div>
              ))}
            </span>
          </span>
        ) : (
          <span key={index}>{segment.text}</span>
        ),
      )}
    </div>
  );
}
