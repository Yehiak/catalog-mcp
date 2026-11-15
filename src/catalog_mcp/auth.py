"""Static bearer-token auth for the Streamable HTTP transport.

A pure ASGI middleware in front of the MCP app: every HTTP request must carry
`Authorization: Bearer <token>`, except GET /healthz. This is deliberately simple
(one shared secret, compared in constant time). For multi-user deployments, put the
server behind a gateway that does OAuth.
"""

from __future__ import annotations

import hmac
import json
import secrets

HEALTH_PATH = "/healthz"


def generate_token() -> str:
    return secrets.token_urlsafe(32)


class BearerTokenMiddleware:
    def __init__(self, app, token: str):
        if not token or len(token) < 16:
            raise ValueError("The HTTP auth token must be at least 16 characters.")
        self.app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] == HEALTH_PATH and scope["method"] == "GET":
            return await _json(send, 200, {"status": "ok"})
        given = dict(scope.get("headers") or []).get(b"authorization", b"")
        if not hmac.compare_digest(given, self._expected):
            return await _json(
                send, 401, {"error": "unauthorized", "detail": "Send 'Authorization: Bearer <token>'."},
                extra=[(b"www-authenticate", b'Bearer realm="catalog-mcp"')],
            )
        return await self.app(scope, receive, send)


async def _json(send, status: int, body: dict, extra: list[tuple[bytes, bytes]] | None = None):
    payload = json.dumps(body).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers + (extra or [])})
    await send({"type": "http.response.body", "body": payload})
