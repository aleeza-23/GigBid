import type { NextConfig } from "next";

/**
 * The API runs as a separate service, and browser calls reach it through
 * `src/app/api/[...path]/route.ts` rather than a `rewrites()` entry.
 *
 * That is deliberate: `rewrites()` is evaluated at build time and its destination is baked
 * into the route manifest, so an `API_BASE_URL` provided at start-up was silently ignored
 * in a production build. The route handler reads it per request, which is what lets one
 * web image be pointed at any API host.
 */
const nextConfig: NextConfig = {
  // `next start` refuses to serve a standalone build, so standalone is opt-in and set only
  // in the container build (infra/docker/Dockerfile.web), which runs server.js directly.
  output: process.env.NEXT_OUTPUT_STANDALONE === "1" ? "standalone" : undefined,
  typedRoutes: false,
};

export default nextConfig;
