/**
 * Typed client over the FastAPI app.
 *
 * The *types* in `schema.d.ts` are generated from the API's OpenAPI document
 * (`npm run generate:api`) and must not be hand-edited, as Part D requires. This file is
 * the thin transport around them: one `request` function, so credentials, errors and the
 * trace header are handled once rather than at thirty call sites.
 *
 * Requests go to `/api/*`, which `next.config.ts` rewrites to the API. That keeps the
 * session cookie first-party — a cross-origin cookie would need `SameSite=None; Secure`,
 * which a local http:// dev server cannot set.
 */

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly traceId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** A 401 means "sign in", which the UI handles differently from a real failure. */
  get isUnauthorized() {
    return this.status === 401;
  }

  /** The marketplace being down is expected and gets a banner, not an error page. */
  get isDegraded() {
    return this.status === 503;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    // The session is an HttpOnly cookie, so every request must carry it.
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });

  const traceId = response.headers.get("X-Trace-Id") ?? undefined;

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? body.error ?? detail;
    } catch {
      // A non-JSON error body (a proxy's HTML 502 page) should not mask the status.
    }
    throw new ApiError(response.status, String(detail), traceId);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

const get = <T>(path: string) => request<T>(path);
const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined });

/* ------------------------------------------------------------------ types */
/* Hand-mirrored from the generated schema for the fields the UI reads. The generated
   `schema.d.ts` is the source of truth; these aliases keep components readable. */

export type Session = {
  user_id: string;
  display_name: string;
  persona: string;
  token: string;
};

export type Health = {
  status: string;
  started_at?: string | null;
  marketplace: Record<string, unknown>;
  persistence: Record<string, unknown>;
  models: Record<string, unknown>;
  retrieval: Record<string, unknown>;
  degraded: string[];
};

export type JobSummary = {
  job_id: string;
  title: string;
  client_id?: string | null;
  category?: string | null;
  skills: string[];
  is_hourly?: boolean | null;
  hourly_low?: number | null;
  hourly_high?: number | null;
  budget?: number | null;
  connects_cost?: number | null;
  fit_coverage?: number | null;
  risk_score?: number | null;
  recommended_action?: "bid" | "watch" | "skip" | null;
  recommended_price?: number | null;
  thread_status?: string | null;
  skill_overlap: number;
};

export type JobPage = {
  total: number;
  limit: number;
  offset: number;
  jobs: JobSummary[];
  marketplace: string;
};

export type RequirementEvidence = {
  requirement: string;
  covered: boolean;
  evidence: Array<Record<string, unknown>>;
  overruled_by_model?: string | null;
};

export type JobDetail = {
  job: Record<string, any>;
  client: Record<string, any>;
  analysed: boolean;
  job_spec?: Record<string, any> | null;
  risk?: Record<string, any> | null;
  fit?: Record<string, any> | null;
  requirement_matrix: RequirementEvidence[];
  price?: Record<string, any> | null;
  decision?: Record<string, any> | null;
  thread_id?: string | null;
  thread_status?: string | null;
  injection_removed: boolean;
};

export type RunSummary = {
  thread_id: string;
  job_id?: string | null;
  status: string;
  trace_id?: string | null;
  decision?: string | null;
  deciding_factor?: string | null;
  fit_coverage?: number | null;
  risk_score?: number | null;
  price_usd?: number | null;
  revision_count: number;
  steps_used: number;
  tokens_used: number;
  budget_exceeded?: string | null;
  unsupported_claims?: number | null;
  awaiting_approval: boolean;
  degraded: Record<string, unknown>;
  trace_entries: number;
};

export type TraceEntry = {
  ts?: string | null;
  node: string;
  kind: string;
  tool?: string | null;
  prompt?: string | null;
  model?: string | null;
  tokens?: number | null;
  duration_ms?: number | null;
  result_summary?: unknown;
  error?: string | null;
};

export type RunDetail = {
  summary: RunSummary;
  trace: TraceEntry[];
  errors: Array<Record<string, unknown>>;
  approval_request?: Record<string, any> | null;
};

export type PendingApproval = {
  thread_id: string;
  kind: "proposal" | "message";
  job_id?: string | null;
  title?: string | null;
  payload: Record<string, any>;
};

export type ApprovalResult = {
  thread_id: string;
  status: string;
  action: string;
  proposal_id?: string | null;
  connects_spent?: number | null;
  idempotent_replay: boolean;
  message_sent: boolean;
  client_reply?: string | null;
  errors: Array<Record<string, unknown>>;
};

export type ThreadSummary = {
  thread_id: string;
  job_id?: string | null;
  client_id?: string | null;
  message_count: number;
  last_from?: string | null;
  last_text?: string | null;
  awaiting_approval: boolean;
  run_status?: string | null;
};

export type MessageThread = {
  thread_id: string;
  job_id?: string | null;
  messages: Array<Record<string, any>>;
  intent?: string | null;
  risk_flags: string[];
  draft_reply?: string | null;
  proposed_slots: Array<Record<string, any>>;
  awaiting_approval: boolean;
  run_status?: string | null;
};

export type PipelineBoard = {
  columns: Record<string, Array<Record<string, any>>>;
  totals: Record<string, number>;
};

export type Analytics = {
  proposals_submitted: number;
  replies: number;
  interviews: number;
  hires: number;
  reply_rate?: number | null;
  hire_rate?: number | null;
  connects_spent: number;
  connects_remaining?: number | null;
  by_category: Array<Record<string, any>>;
  cost: Record<string, any>;
  latency: Record<string, any>;
  top_skip_reasons: Array<Record<string, any>>;
  unsupported_claim_rate?: number | null;
};

export type MemoryView = {
  user_id: string;
  style_rules: Array<Record<string, any>>;
  outcome_rules: Array<Record<string, any>>;
  pending_style_observations: number;
  pending_outcome_observations: number;
  promotion_rules: Array<Record<string, any>>;
};

export type Settings = {
  persona: string;
  autonomy: string;
  weekly_connects_budget?: number | null;
  connects_balance?: number | null;
  hourly_rate_usd?: number | null;
  minimum_project_budget_usd?: number | null;
  marketplace_mode: string;
  model_profile: string;
  retrieval_variant: string;
  embeddings_backend: string;
  embeddings_spec_compliant: boolean;
};

/* ------------------------------------------------------------------- api */

export const api = {
  health: () => get<Health>("/health"),

  login: (username: string, password: string) =>
    post<Session>("/auth/login", { username, password }),
  logout: () => post<{ signed_out: boolean }>("/auth/logout"),
  me: () => get<Session>("/auth/me"),

  jobs: (params: { limit?: number; q?: string; analysed_only?: boolean } = {}) => {
    const search = new URLSearchParams();
    if (params.limit) search.set("limit", String(params.limit));
    if (params.q) search.set("q", params.q);
    if (params.analysed_only) search.set("analysed_only", "true");
    return get<JobPage>(`/jobs?${search.toString()}`);
  },
  job: (jobId: string) => get<JobDetail>(`/jobs/${encodeURIComponent(jobId)}`),

  startRun: (jobId: string, retry = false) =>
    post<RunSummary>("/runs", { job_id: jobId, retry }),
  run: (threadId: string) => get<RunDetail>(`/runs/${encodeURIComponent(threadId)}`),

  approvals: () => get<PendingApproval[]>("/approvals"),
  approval: (threadId: string) =>
    get<PendingApproval>(`/approvals/${encodeURIComponent(threadId)}`),
  decide: (
    threadId: string,
    body: { action: "approve" | "edit" | "reject"; cover_letter?: string; text?: string; reason?: string },
  ) => post<ApprovalResult>(`/approvals/${encodeURIComponent(threadId)}`, body),

  threads: () => get<ThreadSummary[]>("/messages"),
  thread: (threadId: string) =>
    get<MessageThread>(`/messages/${encodeURIComponent(threadId)}`),

  pipeline: () => get<PipelineBoard>("/pipeline"),
  analytics: () => get<Analytics>("/analytics"),
  memory: () => get<MemoryView>("/memory"),
  forget: (kind: "style" | "outcome", key: string, forgetEvidence = false) =>
    post<Record<string, unknown>>("/memory/forget", {
      kind,
      key,
      forget_evidence: forgetEvidence,
    }),
  settings: () => get<Settings>("/settings"),
};

/* ------------------------------------------------------------------- SSE */

export type StreamEvent =
  | { event: "run_started"; data: { thread_id: string } }
  | { event: "node"; data: Record<string, any> }
  | { event: "node_error"; data: Record<string, any> }
  | { event: "awaiting_approval"; data: { thread_id: string; request: Record<string, any> } }
  | { event: "run_finished"; data: Record<string, any> }
  | { event: "stream_error"; data: { error: string } };

/**
 * Stream a run's progress.
 *
 * `fetch` with a reader rather than `EventSource`, for two reasons: EventSource cannot
 * send credentials to a rewritten path reliably, and it cannot issue the POST that
 * `/messages/{id}/draft` needs. The parsing is a few lines, and this way one function
 * serves both streams.
 */
export async function streamRun(
  path: string,
  onEvent: (event: StreamEvent) => void,
  options: { method?: "GET" | "POST"; signal?: AbortSignal } = {},
): Promise<void> {
  const response = await fetch(`/api${path}`, {
    method: options.method ?? "GET",
    credentials: "include",
    headers: { Accept: "text/event-stream" },
    signal: options.signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError(response.status, `stream failed: ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line. Keep any partial frame in the buffer.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      let eventName = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) eventName = line.slice(7).trim();
        else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
        // A line starting with ':' is a keepalive comment; ignore it.
      }
      if (!dataLines.length) continue;
      try {
        onEvent({ event: eventName, data: JSON.parse(dataLines.join("\n")) } as StreamEvent);
      } catch {
        // A frame we cannot parse should not kill the stream.
      }
    }
  }
}
