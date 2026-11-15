import { existsSync } from "node:fs";
import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests against the real stack: GigSim, the API and Next.js, all started here.
 *
 * Three servers rather than mocks, because what these tests are for is the wiring. A
 * mocked approval endpoint would pass while the graph never resumed, which is the one
 * failure Part D exists to prevent.
 *
 * The model is forced offline (`GIGBID_FORCE_OFFLINE`), so drafts are placeholders. That
 * is deliberate: these tests check that approving resumes the graph and spends Connects
 * exactly once, not that a model wrote good prose. A suite whose result depends on a
 * provider's mood is not a suite.
 *
 * A durable SQLite checkpointer is used so the "refresh never loses a pending approval"
 * test means something.
 */

/**
 * Dedicated ports, and no reuse of whatever is already listening.
 *
 * The default `reuseExistingServer: !process.env.CI` silently adopts any server already
 * bound to the port. On a machine with a second Next.js project running on :3000, every
 * spec in this suite failed at sign-in while pointing at that other app's landing page:
 * seven failures that looked like a GigBid bug and were not. Private ports plus
 * `reuseExistingServer: false` make the stack under test the stack that was started here.
 */
const RUN_ID = process.env.E2E_RUN_ID ?? String(Date.now());
process.env.E2E_RUN_ID = RUN_ID;

const PORT_GIGSIM = process.env.E2E_GIGSIM_PORT ?? "8481";
const PORT_API = process.env.E2E_API_PORT ?? "8400";
const PORT_WEB = process.env.E2E_WEB_PORT ?? "3400";

// The specs talk to GigSim directly to check what the marketplace actually received, so
// they need the same port this config started it on.
process.env.E2E_GIGSIM_PORT = PORT_GIGSIM;
process.env.E2E_API_PORT = PORT_API;

/**
 * An absolute, platform-correct interpreter path.
 *
 * Playwright launches webServer commands through the platform shell, and on Windows that
 * is cmd.exe, which cannot resolve `../../.venv/Scripts/python.exe` — it reports
 * "'..' is not recognized as an internal or external command". Resolving the path here
 * keeps the config working on both platforms.
 */
const repoRoot = path.resolve(__dirname, "..", "..");
const venvPython = process.platform === "win32"
  ? path.join(repoRoot, ".venv", "Scripts", "python.exe")
  : path.join(repoRoot, ".venv", "bin", "python");

const PY = process.env.PYTHON_BIN ?? (existsSync(venvPython) ? venvPython : "python");

const sharedEnv = {
  GIGBID_FORCE_OFFLINE: "1",
  EMBED_BACKEND: "lsa",
  RERANK_BACKEND: "lexical",
  GIGSIM_API_TOKEN: "gigsim-dev-token",
  APP_SECRET_KEY: "e2e-secret-not-a-real-key",
  REVIEWER_PASSWORD: "reviewer",
  CHECKPOINTER: "sqlite",
  // Paths are relative to the API's working directory, which is the repository root
  // (`cwd: "../.."` below), not to this file. An earlier `../../data/audit/...` resolved
  // two levels above the repository, so the database was created outside it and survived
  // every "clean" run: threads from a previous suite were already complete, the stream
  // replayed them as finished, and the approval specs reported themselves as skipped.
  //
  // The run id makes each run's database its own file. Durability is tested deliberately,
  // by reloading the page mid-run, never by inheriting another run's approvals, and a file
  // still locked by a server that outlived its run cannot break the next one.
  CHECKPOINT_DB: `data/audit/e2e_checkpoints.${RUN_ID}.sqlite3`,
  MCP_IDEMPOTENCY_DB: `data/audit/e2e_idempotency.${RUN_ID}.sqlite3`,
  PYTHONWARNINGS: "ignore",
  GIGSIM_PORT: PORT_GIGSIM,
  GIGSIM_BASE_URL: `http://127.0.0.1:${PORT_GIGSIM}`,
  // The adapter allows 20 marketplace calls a minute, deliberately below GigSim's 30, so
  // GigBid is never the caller that trips the limit. This suite drives seven user
  // journeys in about four minutes and uses that budget in seconds, and the inbox then
  // correctly shows "marketplace unavailable" instead of jobs. GigSim's own limiter is
  // raised to match, so what is being tested is still a real marketplace call.
  MCP_RATE_LIMIT_PER_MIN: "600",
  GIGSIM_RATE_LIMIT_PER_MIN: "900",
};

export default defineConfig({
  testDir: "./tests/e2e",
  // Wipes the checkpoint and idempotency stores, so a run never inherits approvals from
  // the last one. See tests/e2e/global-setup.ts.
  globalSetup: "./tests/e2e/global-setup.ts",
  // The graph runs real retrieval and several nodes, so a click can legitimately take
  // twenty seconds. Too short a timeout here produces flaky failures that look like bugs.
  // A spec may analyse several jobs before one produces a proposal to approve, and most
  // of the feed is correctly skipped, so the budget is per spec rather than per click.
  timeout: 240_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  // Approving spends Connects from a shared GigSim balance, so the specs must not race
  // each other for it.
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [["html", { open: "never" }], ["list"]],

  use: {
    baseURL: `http://127.0.0.1:${PORT_WEB}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },

  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    // Part D requires the UI to work on a phone, so one project runs there.
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],

  webServer: [
    {
      command: `${PY} -m gigsim`,
      url: `http://127.0.0.1:${PORT_GIGSIM}/health`,
      cwd: "../..",
      reuseExistingServer: false,
      timeout: 60_000,
      env: sharedEnv,
    },
    {
      command: `${PY} -m uvicorn apps.api.main:app --port ${PORT_API} --host 127.0.0.1`,
      url: `http://127.0.0.1:${PORT_API}/health`,
      cwd: "../..",
      reuseExistingServer: false,
      timeout: 90_000,
      env: sharedEnv,
    },
    {
      // `npx next start` directly, not `npm run start --`: npm would append the port
      // after the one already in the script and leave which wins to Next.
      command: `npx next start -p ${PORT_WEB}`,
      url: `http://127.0.0.1:${PORT_WEB}`,
      reuseExistingServer: false,
      timeout: 90_000,
      // The rewrite in next.config.ts is read when the server starts, so the API port has
      // to arrive through the environment rather than the default :8000.
      env: { API_BASE_URL: `http://127.0.0.1:${PORT_API}` },
    },
  ],
});
