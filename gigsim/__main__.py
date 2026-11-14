"""Run the simulator: python -m gigsim"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "gigsim.server:app",
        host=os.getenv("GIGSIM_HOST", "127.0.0.1"),
        port=int(os.getenv("GIGSIM_PORT", "8081")),
        reload=os.getenv("GIGSIM_RELOAD", "0") == "1",
        log_level=os.getenv("GIGSIM_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
