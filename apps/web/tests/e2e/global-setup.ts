import { readdirSync, rmSync } from "node:fs";
import path from "node:path";

/**
 * Remove checkpoint and idempotency stores left by previous runs.
 *
 * Each run writes to files stamped with its own id (see `playwright.config.ts`), so this
 * is housekeeping rather than a precondition: a file still held open by a server that
 * outlived its run cannot be deleted on Windows, and failing the whole suite over that
 * would be worse than leaving a few megabytes behind.
 *
 * The per-run file name is what matters for correctness. Sharing one database between
 * runs meant a thread approved last time replayed as finished, the job was never offered
 * for approval again, and the approval specs reported themselves as skipped: a run that
 * looked green and tested nothing.
 */
const repoRoot = path.resolve(__dirname, "..", "..", "..", "..");
const auditDir = path.join(repoRoot, "data", "audit");

export default function globalSetup() {
  const keep = process.env.E2E_RUN_ID ?? "";
  let entries: string[] = [];
  try {
    entries = readdirSync(auditDir);
  } catch {
    return; // Nothing has written an audit directory yet.
  }

  for (const entry of entries) {
    if (!/^e2e_(checkpoints|idempotency)\./.test(entry)) continue;
    if (keep && entry.includes(keep)) continue;
    try {
      rmSync(path.join(auditDir, entry), { force: true });
    } catch {
      // Held open by a server from an earlier run. The current run uses its own file.
    }
  }
}
