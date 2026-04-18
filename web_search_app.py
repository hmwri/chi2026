#!/usr/bin/env python3
"""Compatibility entrypoint for the unified CHI 2026 MCP and visual app."""
from __future__ import annotations

import uvicorn

from mcp_server import APP_HOST, APP_PORT, mcp

app = mcp.streamable_http_app()


if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
