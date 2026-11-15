"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api, type Health, type Session } from "@/lib/api/client";

const LINKS = [
  { href: "/inbox", label: "Inbox" },
  { href: "/pipeline", label: "Pipeline" },
  { href: "/messages", label: "Messages" },
  { href: "/analytics", label: "Analytics" },
  { href: "/settings", label: "Settings" },
];

/**
 * Top navigation, plus the thing that matters more: the degraded-mode banner.
 *
 * A6 and Part D both require the UI to say when a subsystem is down. The banner lives
 * here rather than per-page so it cannot be forgotten on a screen, and it polls /health
 * so killing the marketplace mid-demo surfaces within thirty seconds without a reload.
 */
export function Nav() {
  const pathname = usePathname();
  const [session, setSession] = useState<Session | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.me().then(setSession).catch(() => setSession(null));
  }, [pathname]);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      api
        .health()
        .then((h) => !cancelled && setHealth(h))
        .catch(() => !cancelled && setHealth(null));
    };
    poll();
    const timer = setInterval(poll, 30_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <>
      <header className="topbar">
        <span className="brand">GigBid</span>
        <nav className="nav" aria-label="Main">
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              aria-current={pathname.startsWith(link.href) ? "page" : undefined}
            >
              {link.label}
            </Link>
          ))}
        </nav>
        {session ? (
          <span className="small dim">
            {session.display_name} · {session.persona}{" "}
            <button
              style={{ minHeight: 32, padding: "4px 10px", marginLeft: 6 }}
              onClick={async () => {
                await api.logout().catch(() => {});
                window.location.href = "/";
              }}
            >
              Sign out
            </button>
          </span>
        ) : (
          <Link href="/" className="small">
            Sign in
          </Link>
        )}
      </header>

      {health && health.degraded.length > 0 && (
        <div style={{ padding: "0 16px", maxWidth: 1180, margin: "12px auto 0", width: "100%" }}>
          <div className="banner banner-warn" role="status">
            <strong>Degraded.</strong>{" "}
            {health.degraded.join(" · ")}
          </div>
        </div>
      )}
    </>
  );
}
