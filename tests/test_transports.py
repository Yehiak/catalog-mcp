"""stdio as a real subprocess, and Streamable HTTP with bearer-token auth."""

import socket
import threading
import time

import httpx2
import pytest
import uvicorn

from catalog_mcp.auth import BearerTokenMiddleware
from catalog_mcp.cli import main
from catalog_mcp.client import http_client, stdio_client
from catalog_mcp.server import create_server

pytestmark = pytest.mark.anyio
TOKEN = "unit-test-token-0123456789"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_url(catalog):
    port = _free_port()
    app = BearerTokenMiddleware(create_server(catalog=catalog).streamable_http_app(host="127.0.0.1"), TOKEN)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_health_needs_no_token(http_url):
    r = httpx2.get(f"{http_url}/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}])
def test_missing_or_wrong_token_is_rejected(http_url, headers):
    r = httpx2.post(f"{http_url}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")


async def test_tool_call_over_http(http_url):
    async with http_client(f"{http_url}/mcp", TOKEN) as c:
        r = await c.call_tool("search_products", {"query": "racing wheel", "mode": "keyword", "limit": 2})
        assert not r.is_error
        assert r.structured_content["results"][0]["product_id"] == "B0GAME0006"


async def test_tool_call_over_stdio(catalog_dir):
    # The subprocess opens the fixture catalog itself; keyword search needs no embedding model.
    async with stdio_client(catalog_dir, extra_args=["--no-warm-up"]) as c:
        tools = {t.name for t in (await c.list_tools()).tools}
        assert "search_products" in tools
        r = await c.call_tool("search_products", {"query": "DC31-00054A", "mode": "keyword", "limit": 1})
        assert r.structured_content["results"][0]["product_id"] == "B0APPL0008"


def test_short_token_rejected():
    with pytest.raises(ValueError):
        BearerTokenMiddleware(lambda *a: None, "short")


def test_http_requires_token(monkeypatch, tmp_path):
    monkeypatch.delenv("CATALOG_MCP_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="needs a token"):
        main(["serve", "--transport", "http", "--data-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="loopback"):
        main(["serve", "--transport", "http", "--host", "0.0.0.0", "--no-auth", "--data-dir", str(tmp_path)])
