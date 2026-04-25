"""HTTP route: GET /private-pipeline/version

Lets the GIMP plug-in detect what the private downstream distribution-specific server-side
features are available, so it can opt into the encrypted-input /
encrypted-output workflow paths only when the pack is present.

Response shape:
{
  "private_pack": "0.1.0",
  "auth_token_present": true,            # bool — server can derive keys
  "wire_v1": true,                       # wire_envelope module loaded
  "nodes": ["PrivateDecryptLoadImage", "PrivateEncryptSaveImage"],
  "features": {
    "encrypted_input":  true,
    "encrypted_output": true
  }
}

The GIMP plug-in's `_inline.probe` action already checks for inline
transport (ETN_LoadImageBase64); this is its the private downstream distribution-specific
companion.
"""
from __future__ import annotations

import os
from pathlib import Path

PACK_VERSION = "0.1.0"


def _auth_token_present() -> bool:
    if (Path.home() / ".spellcaster" / "auth_token").is_file():
        return True
    if os.environ.get("PRIVATE_AUTH_TOKEN", "").strip():
        return True
    return False


def _build_status() -> dict:
    try:
        from .private-pipeline_privacy import wire_envelope as _we
        wire_v1 = True
        wire_using_pure = bool(getattr(_we, "_USING_PURE_FALLBACK", False))
    except Exception as exc:
        wire_v1 = False
        wire_using_pure = False
    auth_ok = _auth_token_present()
    return {
        "private_pack": PACK_VERSION,
        "auth_token_present": auth_ok,
        "wire_v1": wire_v1,
        "wire_using_pure_fallback": wire_using_pure,
        "nodes": [
            "PrivateDecryptLoadImage",
            "PrivateEncryptSaveImage",
        ],
        "features": {
            "encrypted_input":  wire_v1 and auth_ok,
            "encrypted_output": wire_v1 and auth_ok,
        },
    }


def _register_routes() -> bool:
    """Attach the version route to ComfyUI's PromptServer. Returns True
    on success; False when not running inside ComfyUI."""
    try:
        from server import PromptServer  # ComfyUI singleton
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

    @routes.get("/private-pipeline/version")
    async def _version(request):
        return web.json_response(_build_status())

    return True


def is_available() -> bool:
    return _register_routes()


__all__ = ["PACK_VERSION", "is_available", "_register_routes"]
