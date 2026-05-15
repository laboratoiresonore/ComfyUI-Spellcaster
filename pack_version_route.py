"""HTTP route: GET /spellcaster/version

Unified version + capability handshake. Single round-trip surface that
external caps servers (Voodoomaster's `/v1/capabilities` composer, the
Wizard Guild's startup probe) can hit to confirm:

  * which version of the ComfyUI-Spellcaster pack is loaded
  * which optional subsystems registered routes (presence broker,
    blob bus, privacy cleanup, private crypto add-on)
  * which security postures are active (localhost-only mode, at-rest
    encryption availability for the bus)

This is a deliberate consolidation of the per-subsystem version /
status hooks that used to live in `__init__.py`'s startup print line
+ the per-route `/spellcaster/private/version` endpoint. Both still
exist for backwards compatibility; the unified endpoint is what new
callers (caps_server) should bind against.

Response shape:
    {
      "pack":           "ComfyUI-Spellcaster",
      "pack_version":   "<git-describe-or-static>",
      "schema_version": 1,
      "subsystems": {
        "presence":         true,
        "blob_bus":         true,
        "privacy_cleanup":  true,
        "model_repair":     true,
        "private_crypto":   true
      },
      "security": {
        "localhost_only":     false,
        "blob_at_rest_aead":  true,
        "blob_one_shot":      true,
        "blob_encrypt_param": true
      },
      "nodes": ["SpellcasterLoader", ...],
      "routes": ["/spellcaster/blob/put", ...]
    }

Schema-version bumps signal breaking changes (field rename, semantic
shift). Adding new keys is forward-compatible — clients that ignore
unknowns just don't gate on the new feature. Same convention as the
caps doc.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Static fallback if no git history is available (e.g. installed via
# zip download). Bump in lockstep with material protocol changes —
# CI's mirror-drift check uses this to detect a stale install.
PACK_VERSION_STATIC = "0.2.0"


def _git_describe() -> str | None:
    """Best-effort ``git describe --always --dirty`` against the pack
    dir. Returns None on any failure (no git binary, no .git, network
    detached, repo permissions). Cheap (~10 ms warm)."""
    here = Path(__file__).resolve().parent
    try:
        proc = subprocess.run(
            ["git", "-C", str(here), "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=5,
            check=False)
        v = proc.stdout.strip()
        return v or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _resolve_pack_version() -> str:
    """Resolution order: git-describe > static. Future: read a baked
    `_version.py` like Voodoomaster does for PyInstaller bundles."""
    v = _git_describe()
    if v:
        return v
    return PACK_VERSION_STATIC


def _build_manifest() -> dict[str, Any]:
    """Compose the version document. Imports are deferred + best-effort
    — a missing optional module just means the subsystem reports False
    rather than aborting the whole endpoint."""
    # Subsystem availability — read the cached `is_available()` flag
    # from each module rather than re-registering routes.
    subsystems: dict[str, bool] = {
        "presence": False,
        "blob_bus": False,
        "privacy_cleanup": False,
        "model_repair": False,
        "private_crypto": False,
    }
    security: dict[str, Any] = {
        "localhost_only": False,
        "blob_at_rest_aead": False,
        "blob_one_shot": True,
        "blob_encrypt_param": True,
    }
    nodes: list[str] = []

    try:
        from . import presence as _p
        subsystems["presence"] = bool(_p.is_available())
    except Exception:
        pass
    try:
        from . import blob_bus as _b
        subsystems["blob_bus"] = bool(_b.is_available())
        security["localhost_only"] = bool(
            getattr(_b, "LOCALHOST_ONLY", False))
        # The at-rest AEAD path is available iff the bus can resolve
        # its key (which means the auth token is configured AND
        # wire_envelope is importable).
        try:
            security["blob_at_rest_aead"] = (
                _b._resolve_bus_key() is not None)
        except Exception:
            security["blob_at_rest_aead"] = False
    except Exception:
        pass
    try:
        from . import privacy_cleanup as _pc
        # privacy_cleanup re-registers on every is_available() call —
        # idempotent thanks to aiohttp dedupe but we'd rather not poke
        # it from here. Treat presence of the module as the signal.
        subsystems["privacy_cleanup"] = hasattr(_pc, "_handle")
    except Exception:
        pass
    try:
        from . import model_repair as _mr  # noqa: F401
        subsystems["model_repair"] = True
    except Exception:
        pass
    try:
        from . import private_crypto_nodes as _pcn
        subsystems["private_crypto"] = bool(
            getattr(_pcn, "NODE_CLASS_MAPPINGS", {}))
    except Exception:
        pass

    # Node class names — small list; useful for clients that want to
    # confirm a specific node is registered without paging through
    # /object_info.
    try:
        from . import __init__ as _self  # type: ignore
    except Exception:
        _self = None
    if _self is None:
        try:
            # When this module is imported by the pack, __init__ has
            # already populated NODE_CLASS_MAPPINGS — grab via the
            # parent package.
            import sys
            parent = sys.modules.get(__name__.rsplit(".", 1)[0])
            if parent is not None:
                nodes = sorted(getattr(parent, "NODE_CLASS_MAPPINGS", {}).keys())
        except Exception:
            nodes = []

    routes = [
        "/spellcaster/presence/register",
        "/spellcaster/presence/heartbeat",
        "/spellcaster/presence/list",
        "/spellcaster/presence/unregister",
        "/spellcaster/blob/put",
        "/spellcaster/blob/list",
        "/spellcaster/blob/{hash}",
        "/spellcaster/privacy/delete",
        "/spellcaster/private/version",
        "/spellcaster/private/setup",
        "/spellcaster/version",
    ]

    return {
        "pack": "ComfyUI-Spellcaster",
        "pack_version": _resolve_pack_version(),
        "schema_version": SCHEMA_VERSION,
        "subsystems": subsystems,
        "security": security,
        "nodes": nodes,
        "routes": routes,
    }


def _register_routes() -> bool:
    """Attach GET /spellcaster/version to ComfyUI's PromptServer.
    Returns True on success, False outside a ComfyUI runtime."""
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

    @routes.get("/spellcaster/version")
    async def _version(_request):
        return web.json_response(_build_manifest())

    return True


def is_available() -> bool:
    return _register_routes()


__all__ = [
    "PACK_VERSION_STATIC",
    "SCHEMA_VERSION",
    "_build_manifest",
    "is_available",
]
