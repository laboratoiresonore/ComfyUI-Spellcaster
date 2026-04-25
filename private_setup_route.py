"""POST /private-pipeline/setup — single-use auth token receiver.

The the private downstream distribution GIMP plugin posts its auth token here ONCE on first
contact with a freshly-installed pack. The token gets written to
<pack_dir>/.auth_token (where _resolve_auth_token() finds it) and a
.setup_done marker is created.

Subsequent POSTs are refused (409 Conflict) until an admin manually
deletes <pack_dir>/.setup_done. This prevents a malicious LAN peer
from overwriting the auth token after first install.

Body shape:
    {"auth_token": "<43-char base64url string from client>"}

Responses:
    200  {"ok": true,  "note": "token saved"}
    409  {"ok": false, "error": "already configured"}
    400  {"ok": false, "error": "<reason>"}
"""
from __future__ import annotations

import json
import os
from pathlib import Path


_HERE = Path(__file__).resolve().parent
TOKEN_PATH    = _HERE / ".auth_token"
SETUP_MARKER  = _HERE / ".setup_done"


def _handle(body: dict) -> tuple[int, dict]:
    if SETUP_MARKER.is_file():
        return 409, {"ok": False,
                     "error": ("private-pipeline pack is already configured. "
                                "Delete .setup_done from the pack dir to "
                                "re-provision.")}
    token = (body.get("auth_token") or "").strip()
    if not token:
        return 400, {"ok": False, "error": "missing auth_token"}
    if len(token.encode("utf-8")) < 16:
        return 400, {"ok": False,
                     "error": "auth_token too short (need >=16 bytes)"}
    try:
        TOKEN_PATH.write_text(token, encoding="utf-8")
        try:
            os.chmod(TOKEN_PATH, 0o600)
        except OSError:
            pass
        SETUP_MARKER.write_text("1", encoding="utf-8")
    except Exception as exc:
        return 400, {"ok": False,
                     "error": f"write failed: {type(exc).__name__}: {exc}"}
    return 200, {"ok": True,
                 "note": (f"token saved to {TOKEN_PATH}; restart ComfyUI "
                          f"to activate the encrypted pipeline")}


def _register_routes() -> bool:
    try:
        from server import PromptServer
    except Exception:
        return False
    try:
        from aiohttp import web
    except Exception:
        return False
    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    routes = getattr(instance, "routes", None)
    if routes is None:
        return False

    @routes.post("/private-pipeline/setup")
    async def _setup(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        status, payload = _handle(body)
        return web.json_response(payload, status=status)

    return True


def is_available() -> bool:
    return _register_routes()


__all__ = ["is_available", "_handle", "TOKEN_PATH", "SETUP_MARKER"]
