"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api/client";
import { ErrorState } from "@/components/States";

/**
 * Sign-in and onboarding.
 *
 * Multi-user: the session scopes every thread id, so two freelancers bidding on the same
 * job never share state. The reviewer account is seeded server-side from the environment,
 * which is how the deployed link is opened with credentials submitted privately.
 */
export default function SignInPage() {
  const router = useRouter();
  const [username, setUsername] = useState("reviewer");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  // Already signed in? Go straight to the inbox.
  useEffect(() => {
    api
      .me()
      .then(() => router.replace("/inbox"))
      .catch(() => {});
  }, [router]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.login(username, password);
      router.push("/inbox");
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ maxWidth: 400, margin: "40px auto" }}>
      <h1>Sign in to GigBid</h1>
      <p className="subtitle">
        A team of agents finds work, judges it and drafts proposals. You approve everything
        before it leaves.
      </p>

      {error instanceof ApiError && error.status === 401 ? (
        <div className="banner banner-danger" role="alert">
          Incorrect username or password.
        </div>
      ) : error ? (
        <ErrorState error={error} />
      ) : null}

      <form onSubmit={submit} className="card">
        <div className="field">
          <label htmlFor="username">Username</label>
          <input
            id="username"
            name="username"
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button className="btn-primary" type="submit" disabled={busy} style={{ width: "100%" }}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
