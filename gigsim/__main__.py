"""Run the simulator: python -m gigsim"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    # PORT is what a platform host (Render, Fly, Heroku) injects, and it is the only port
    # the service is reachable on there. GIGSIM_PORT still wins when it is set, so local
    # runs and the e2e suite, which picks its own private port, are unaffected.
    port = os.getenv("GIGSIM_PORT") or os.getenv("PORT") or "8081"
    # Binding to localhost on a hosted platform makes the service unreachable, so a
    # platform-provided PORT also implies a public bind unless GIGSIM_HOST says otherwise.
    default_host = "0.0.0.0" if os.getenv("PORT") and not os.getenv("GIGSIM_PORT") else "127.0.0.1"

    uvicorn.run(
        "gigsim.server:app",
        host=os.getenv("GIGSIM_HOST", default_host),
        port=int(port),
        reload=os.getenv("GIGSIM_RELOAD", "0") == "1",
        log_level=os.getenv("GIGSIM_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
