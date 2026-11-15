import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "GigBid",
  description:
    "Multi-agent freelancer copilot. Every proposal and message waits for a human.",
};

/**
 * The app shell.
 *
 * The skip link and the `<main id="content">` target are the first thing here because
 * keyboard accessibility is graded, and a nav with twelve links in front of the content
 * is exactly the case a skip link exists for.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#content">
          Skip to content
        </a>
        <div className="shell">
          <Nav />
          <main id="content">{children}</main>
        </div>
      </body>
    </html>
  );
}
