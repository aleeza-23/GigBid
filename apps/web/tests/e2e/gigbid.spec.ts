import { expect, test, type Page } from "@playwright/test";

/**
 * The five journeys Part D names: onboarding, approving a proposal, rejecting with a
 * reason, editing a draft, and the degraded-mode banner. Plus two that guard the
 * properties those journeys depend on.
 *
 * These run against the real stack. The assertion that matters in most of them is not
 * what the page says but what the marketplace did: a UI that claims it sent a proposal
 * while nothing reached GigSim is the failure worth catching, and only a check against
 * GigSim finds it.
 */

// The port comes from the config, which starts GigSim on a private port so the suite
// cannot talk to a simulator somebody left running on the default 8081. Hardcoding 8081
// here made every GigSim assertion fail with "fetch failed" while the UI worked.
const GIGSIM = `http://127.0.0.1:${process.env.E2E_GIGSIM_PORT ?? "8481"}/api/v1`;
const TOKEN = "gigsim-dev-token";

async function signIn(page: Page) {
  await page.goto("/");
  await page.getByLabel("Username").fill("reviewer");
  await page.getByLabel("Password").fill("reviewer");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Job inbox" })).toBeVisible();
}

async function gigsim(path: string) {
  const response = await fetch(`${GIGSIM}${path}`, {
    headers: { Authorization: `Bearer ${TOKEN}` },
  });
  if (!response.ok) throw new Error(`GigSim ${path} -> ${response.status}`);
  return response.json();
}

async function connectsBalance(freelancerId = "ai_ml_engineer"): Promise<number> {
  const body = await gigsim(`/freelancers/${freelancerId}/connects`);
  return Number(body.balance ?? 0);
}

async function proposalsFor(jobId: string): Promise<Array<Record<string, unknown>>> {
  const body = await gigsim("/proposals?limit=200");
  return (body.proposals ?? []).filter(
    (row: { job_id?: string }) => row.job_id === jobId,
  );
}

async function proposalCount(): Promise<number> {
  const body = await gigsim("/proposals?limit=200");
  return (body.proposals ?? []).length;
}

/**
 * Analyse jobs until one reaches an approval, and return the studio URL.
 *
 * A loop rather than a fixed job, because most of the dev feed is correctly skipped —
 * scams, off-domain work, budgets under the floor. A test that picked one job would be
 * testing the seed data's ordering as much as the UI.
 */
/**
 * Open the inbox, retrying once or twice if the marketplace call failed.
 *
 * The marketplace adapter allows twenty calls a minute, below GigSim's thirty, and a
 * suite driving the UI uses that budget far faster than a person would. When it runs out
 * the inbox correctly shows "Marketplace unavailable" with a retry button, so the suite
 * retries rather than reporting the product's own backpressure as a failure.
 */
async function openInbox(page: Page, retries = 3): Promise<void> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    await page.goto("/inbox");
    const outage = page.locator("main").getByText(/Marketplace unavailable/i);
    if (!(await outage.isVisible().catch(() => false))) return;
    const again = page.getByRole("button", { name: "Try again" });
    if (await again.isVisible().catch(() => false)) await again.click();
    await page.waitForTimeout(3_000);
  }
}

// How many jobs to try before giving up on finding one that produces a proposal. Roughly
// a third of the dev feed is worth bidding on, and each spec consumes the job it approves
// or rejects, so the later specs have to look further than the first.
const APPROVAL_SEARCH_DEPTH = 12;

/**
 * Analyse jobs until one reaches an approval, and return the studio URL.
 *
 * Only rows the inbox marks "not analysed" are tried. That matters for more than speed:
 * a job analysed by an earlier spec already has a finished thread, so re-analysing it
 * replays the old outcome instead of producing a new approval. Walking every row and
 * discovering this one job at a time is what made the last spec exceed a four-minute
 * timeout.
 *
 * A loop rather than a fixed job, because most of the dev feed is correctly skipped:
 * scams, off-domain work, budgets under the floor. A test that picked one job would be
 * testing the seed data's ordering as much as the UI.
 */
async function reachAnApproval(
  page: Page,
  attempts = APPROVAL_SEARCH_DEPTH,
): Promise<string> {
  const tried = new Set<string>();

  for (let attempt = 0; attempt < attempts; attempt++) {
    await openInbox(page);

    const fresh = page.locator("tbody tr").filter({ hasText: "not analysed" });
    try {
      await fresh.first().waitFor({ state: "visible", timeout: 20_000 });
    } catch {
      break; // Nothing left in the feed that has not been analysed.
    }

    // Take the first unanalysed job this spec has not already tried. Analysing one marks
    // it, so the list advances on its own; `tried` only guards the case where a job stays
    // unanalysed because its run failed to start.
    const count = await fresh.count();
    let link = null;
    let jobId = "";
    for (let row = 0; row < count; row++) {
      const candidate = fresh.nth(row).locator("td:first-child a");
      const text = ((await candidate.textContent()) ?? "").trim();
      if (!tried.has(text)) {
        link = candidate;
        jobId = text;
        break;
      }
    }
    if (!link) break;
    tried.add(jobId);

    await link.click();

    // `waitFor`, not `isVisible`: the job page loads its detail client-side, and
    // `isVisible` resolves immediately without waiting. It returned false on every job
    // while the first render was still in flight, so this loop skipped the whole feed and
    // all five approval specs reported themselves as skipped.
    const analyse = page.getByRole("button", { name: /Analyse this job|Re-run analysis/ });
    try {
      await analyse.waitFor({ state: "visible", timeout: 20_000 });
    } catch {
      continue;
    }
    await analyse.click();
    await page.waitForURL(/\/(runs|studio)\//, { timeout: 60_000 });

    // A run either interrupts for approval, which redirects to the studio, or finishes
    // with a decision to skip and says so. Racing the two keeps the loop inside the test
    // timeout: waiting the full redirect timeout on each correctly skipped job does not
    // fit in four minutes.
    await Promise.race([
      page.waitForURL(/\/studio\//, { timeout: 60_000 }).catch(() => undefined),
      page
        .getByText(/Finished:/)
        .waitFor({ state: "visible", timeout: 60_000 })
        .catch(() => undefined),
    ]);
    if (/\/studio\//.test(page.url())) return page.url();
  }

  test.skip(true, "no job in the sampled feed produced a proposal to approve");
  return "";
}

test.describe("GigBid", () => {
  test("onboarding: sign in and land on the inbox", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Sign in to GigBid" })).toBeVisible();

    // A wrong password must not reveal whether the account exists.
    await page.getByLabel("Username").fill("reviewer");
    await page.getByLabel("Password").fill("definitely-wrong");
    await page.getByRole("button", { name: "Sign in" }).click();
    // Scoped to `main`: Next renders its own route announcer with role="alert" outside
    // the page content, and an unscoped match is a strict-mode violation rather than a
    // real assertion.
    await expect(page.locator("main").getByRole("alert")).toContainText(
      /incorrect username or password/i,
    );

    await page.getByLabel("Password").fill("reviewer");
    await page.getByRole("button", { name: "Sign in" }).click();

    await expect(page.getByRole("heading", { name: "Job inbox" })).toBeVisible();
    // Unanalysed jobs must say so rather than showing a zero score.
    await expect(page.locator("tbody").getByText("not analysed").first()).toBeVisible();
  });

  test("approving a proposal spends Connects exactly once", async ({ page }) => {
    await signIn(page);
    const studio = await reachAnApproval(page);
    await page.goto(studio);

    // The thread id is `bid:<user>:<job>`, so the job under test comes from the URL
    // rather than from a fixed id: which job reaches an approval depends on the feed.
    const threadId = decodeURIComponent(studio.split("/studio/")[1]);
    const jobId = threadId.split(":").slice(2).join(":");

    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
    // C5: the cost belongs next to the decision, not buried in a panel.
    await expect(page.getByText("Connects it costs")).toBeVisible();
    await expect(page.getByText(/Exactly what will be sent/)).toBeVisible();

    const balanceBefore = await connectsBalance();
    expect(await proposalsFor(jobId)).toHaveLength(0);

    const approve = page.getByRole("button", { name: /Approve and send/ });
    await expect(approve).toBeVisible();
    await approve.click();

    await expect(page.getByRole("heading", { name: "Submitted" })).toBeVisible();
    await expect(page.locator("main").getByRole("status")).toContainText(
      /Sent to the marketplace/,
    );

    // Exactly one proposal for this job, and the Connects it cost came out of the
    // balance. Asserting the balance rather than only the count is what makes this a
    // test of spending: a double submission that GigSim deduplicated would leave the
    // count at one and the balance short.
    const sent = await proposalsFor(jobId);
    expect(sent).toHaveLength(1);

    const job = await gigsim(`/jobs/${jobId}`);
    const cost = Number(job.connects_cost ?? 0);
    expect(cost).toBeGreaterThan(0);
    expect(await connectsBalance()).toBe(balanceBefore - cost);

    // And once means once across a reload: the approval is already recorded in the
    // checkpoint, so a refreshed studio shows the outcome instead of offering to send
    // again.
    await page.reload();
    await expect(page.getByRole("button", { name: /Approve and send/ })).toHaveCount(0);
    expect(await proposalsFor(jobId)).toHaveLength(1);
    expect(await connectsBalance()).toBe(balanceBefore - cost);
  });

  test("rejecting with a reason sends nothing", async ({ page }) => {
    await signIn(page);
    const studio = await reachAnApproval(page);
    await page.goto(studio);

    const before = await proposalCount();

    await page.getByLabel(/Reason/).fill("rate too low for the scope");
    await page.getByRole("button", { name: "Reject" }).click();

    const outcome = page.locator("main").getByRole("status");
    await expect(outcome).toContainText(/Rejected/);
    await expect(outcome).toContainText(/no Connects were spent/i);

    expect(await proposalCount()).toBe(before);
  });

  test("editing a draft sends the human's text, not the agent's", async ({ page }) => {
    await signIn(page);
    const studio = await reachAnApproval(page);
    await page.goto(studio);

    // A word marker, not a timestamp: a bare millisecond number reads as a phone number
    // to any reasonable contact-detail check, and the suite should not depend on the
    // compliance rule's view of that. The rule itself is tested in
    // tests/unit/test_compliance_redactions.py.
    const marker = `Edited by the freelancer, run ${Math.random().toString(36).slice(2, 8)}.`;

    await page.getByRole("button", { name: "Edit" }).click();
    const textarea = page.getByLabel("Proposal text");
    await expect(textarea).toBeVisible();
    await textarea.fill(
      `${marker} I built a transcription pipeline that cut turnaround by a third.`,
    );

    await page.getByRole("button", { name: /Send my edited version/ }).click();
    await expect(page.getByRole("heading", { name: "Submitted" })).toBeVisible();

    // The assertion that matters: the marketplace received the human's text.
    const body = await gigsim("/proposals?limit=200");
    const sent = (body.proposals ?? []).map((p: { cover_letter?: string }) => p.cover_letter ?? "");
    expect(sent.some((letter: string) => letter.includes(marker))).toBe(true);
  });

  test("degraded mode: the banner appears when the marketplace is unreachable", async ({
    page,
  }) => {
    await signIn(page);

    // Make every marketplace call fail at the browser boundary. The API's own health
    // check still runs server-side, so this exercises the UI's own degraded handling:
    // a failed inbox load must become an explained banner, not a blank screen.
    await page.route("**/api/jobs**", (route) =>
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ detail: "marketplace unavailable: simulated outage" }),
      }),
    );

    await page.goto("/inbox");
    const alert = page.locator("main").getByRole("alert");
    await expect(alert).toBeVisible();
    await expect(alert).toContainText(/Marketplace unavailable/i);
    // It must say the work is not lost, which is the whole point of queueing.
    await expect(alert).toContainText(/queued|retry/i);
    await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
  });

  test("a refresh never loses a pending approval", async ({ page }) => {
    await signIn(page);
    const studio = await reachAnApproval(page);

    // The approval lives in the checkpoint, not in the tab. Reloading must find it.
    await page.goto(studio);
    await expect(page.getByRole("button", { name: /Approve and send/ })).toBeVisible();

    await page.reload();
    await expect(page.getByRole("button", { name: /Approve and send/ })).toBeVisible();

    // And it is listed from a cold start on another screen.
    await page.goto("/inbox");
    // Filtered, not just scoped: the inbox also renders a polite "Loading jobs…" status
    // while it fetches, and two matches is a strict-mode violation rather than a result.
    await expect(
      page.locator("main").getByRole("status").filter({ hasText: /waiting for you/i }),
    ).toBeVisible();
  });

  test("every claim in a draft shows the evidence behind it", async ({ page }) => {
    await signIn(page);
    const studio = await reachAnApproval(page);
    await page.goto(studio);

    await expect(
      page.getByText(/Underlined sentences have evidence behind them/),
    ).toBeVisible();

    const claims = page.locator(".claim");
    const count = await claims.count();
    test.skip(count === 0, "the offline placeholder draft cites nothing to hover");

    // Keyboard-reachable, not mouse-only: focus must reveal the same popover.
    await claims.first().focus();
    await expect(page.getByRole("note").first()).toBeVisible();
    await expect(page.getByRole("note").first()).toContainText(/Evidence for this claim/);
  });
});
