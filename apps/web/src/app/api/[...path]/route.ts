/**
 * One-origin proxy to the FastAPI app.
 *
 * This existed as a `rewrites()` entry in next.config.ts, which was wrong in a way that
 * only showed up in a production build: Next evaluates `rewrites()` at *build* time and
 * bakes the destination into the route manifest, so `API_BASE_URL` set at start-up was
 * ignored and every request went to the compiled-in :8000. The e2e suite saw the resulting
 * proxy failure as a 500 and reported "Internal Server Error" on a wrong password, which
 * looks like an auth bug and is not one.
 *
 * A route handler resolves the target per request, so the same web image can be pointed at
 * a different API host by changing one environment variable -- which is what the container
 * deployment needs, since the API is a separate service there.
 *
 * Why proxy at all rather than call the API cross-origin: the session is an HttpOnly
 * cookie, and a cross-site cookie needs `SameSite=None; Secure`, which a local http://
 * server cannot set. Same origin keeps it first-party.
 */

const API_BASE = (process.env.API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

// Hop-by-hop and length headers: forwarding them lets the client's framing contradict ours.
const STRIP_REQUEST = ["host", "connection", "content-length", "accept-encoding"];
const STRIP_RESPONSE = ["content-length", "content-encoding", "transfer-encoding", "connection"];

async function proxy(request: Request, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const incoming = new URL(request.url);
  const target = `${API_BASE}/${(path ?? []).join("/")}${incoming.search}`;

  const headers = new Headers(request.headers);
  for (const name of STRIP_REQUEST) headers.delete(name);

  const hasBody = request.method !== "GET" && request.method !== "HEAD";

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body: hasBody ? request.body : undefined,
      // Required by undici whenever the body is a stream rather than a buffer.
      ...(hasBody ? { duplex: "half" } : {}),
      redirect: "manual",
      cache: "no-store",
    } as RequestInit);
  } catch (error) {
    // The API being down is an expected state the UI has a banner for, so it is reported
    // as 503 rather than as a proxy crash the client cannot interpret.
    return Response.json(
      { detail: `API unreachable at ${API_BASE}: ${(error as Error).message}` },
      { status: 503 },
    );
  }

  const out = new Headers();
  upstream.headers.forEach((value, name) => {
    if (!STRIP_RESPONSE.includes(name.toLowerCase()) && name.toLowerCase() !== "set-cookie") {
      out.set(name, value);
    }
  });
  // Set-Cookie can appear more than once and must not be folded into one comma-joined
  // header, which browsers reject. `getSetCookie` is the only API that keeps them apart.
  for (const cookie of upstream.headers.getSetCookie?.() ?? []) out.append("set-cookie", cookie);

  // The body is passed through unread so an SSE stream stays a stream: the run timeline
  // depends on tokens arriving as they are produced, not after the response completes.
  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: out,
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const HEAD = proxy;
export const OPTIONS = proxy;

// Nothing here may be cached or statically analysed: every call depends on a cookie.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";
