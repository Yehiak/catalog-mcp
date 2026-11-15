"""Small helpers to connect an MCP client to this server over each transport.

Used by the tests and the benchmark scripts; handy for your own scripts too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client


def stdio_client(
    data_dir: str | Path | None = None,
    command: list[str] | None = None,
    env: dict | None = None,
    extra_args: list[str] | None = None,
) -> Client:
    """Launch `catalog-mcp serve` as a subprocess and talk to it over stdin/stdout."""
    cmd = command or [sys.executable, "-m", "catalog_mcp.cli"]
    args = [*cmd[1:], "serve", "--transport", "stdio"]
    if data_dir is not None:
        args += ["--data-dir", str(data_dir)]
    args += extra_args or []
    # The SDK passes only a minimal environment to the child by default; keep ours so
    # HF_HOME and friends still point at the local model cache.
    return Client(StdioServerParameters(command=cmd[0], args=args, env={**os.environ, **(env or {})}))


def http_client(url: str, token: str | None = None) -> Client:
    """Connect to a running Streamable HTTP server, sending the bearer token if given."""
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return Client(streamable_http_client(url, http_client=create_mcp_http_client(headers=headers)))
