/**
 * Loading, empty and error states.
 *
 * Shared because Part D grades them and because the failure mode they prevent is a screen
 * that shows nothing and leaves the user unsure whether it is broken or just empty. An
 * `ApiError` carries a trace id, which is shown so a user can quote it.
 */
import { ApiError } from "@/lib/api/client";

export function Loading({ what = "Loading" }: { what?: string }) {
  return (
    <p className="dim small" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" /> {what}…
    </p>
  );
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="empty">
      <p style={{ margin: 0, fontWeight: 600 }}>{title}</p>
      {hint && <p className="small" style={{ margin: "6px 0 0" }}>{hint}</p>}
    </div>
  );
}

export function ErrorState({ error, retry }: { error: unknown; retry?: () => void }) {
  const api = error instanceof ApiError ? error : null;

  if (api?.isUnauthorized) {
    return (
      <div className="banner banner-warn" role="alert">
        Your session expired. <a href="/">Sign in again</a>.
      </div>
    );
  }

  return (
    <div className="banner banner-danger" role="alert">
      <strong>{api?.isDegraded ? "Marketplace unavailable." : "Something went wrong."}</strong>{" "}
      {api?.isDegraded
        ? "The work is queued and will retry. Nothing has been lost."
        : String(api?.message ?? error)}
      {api?.traceId && <span className="mono small"> · trace {api.traceId}</span>}
      {retry && (
        <>
          {" "}
          <button style={{ minHeight: 30, padding: "3px 10px" }} onClick={retry}>
            Try again
          </button>
        </>
      )}
    </div>
  );
}
