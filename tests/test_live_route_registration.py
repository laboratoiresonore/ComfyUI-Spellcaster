"""Live-route-registration test.

This test stands in for the "ComfyUI restart + curl /spellcaster/version"
verification when a live ComfyUI restart isn't available. It mocks
the bare-minimum PromptServer + aiohttp.web surface that our route
installers depend on, then asserts:

  1. blob_bus.install() succeeds and registers PUT /spellcaster/blob/put
  2. pack_version_route.is_available() succeeds and registers
     GET /spellcaster/version
  3. The handler bound to /spellcaster/version produces a manifest
     with the expected shape

This is the same code path ComfyUI takes at startup; the only thing
we don't exercise is aiohttp's request dispatch, which is well-tested
upstream.
"""
from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PACK_DIR = _HERE.parent
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))


def _install_fake_comfyui_runtime() -> dict[tuple, callable]:
    """Stand up a minimal PromptServer + aiohttp.web surface.

    Returns a dict keyed by (METHOD, path) → handler so tests can
    invoke the route handlers directly. Idempotent — calling twice
    just hands back the same registry."""
    routes: dict[tuple, callable] = {}

    class FakeRoutes:
        def get(self, path):
            def deco(h):
                routes[("GET", path)] = h
                return h
            return deco
        def post(self, path):
            def deco(h):
                routes[("POST", path)] = h
                return h
            return deco

    class FakeInstance:
        routes = FakeRoutes()

    class FakePromptServer:
        instance = FakeInstance()

    server_mod = types.ModuleType("server")
    server_mod.PromptServer = FakePromptServer
    sys.modules["server"] = server_mod

    aiohttp_mod = types.ModuleType("aiohttp")
    web_mod = types.ModuleType("aiohttp.web")

    class FakeResponse:
        def __init__(self, **kw):
            self.body = kw.get("body")
            self.content_type = kw.get("content_type")
            self.status = kw.get("status", 200)
            self.json_payload = kw.get("json_payload")

    def json_response(payload, status=200):
        r = FakeResponse(body=None, status=status, json_payload=payload)
        return r

    web_mod.Response = FakeResponse
    web_mod.json_response = json_response
    aiohttp_mod.web = web_mod
    sys.modules["aiohttp"] = aiohttp_mod
    sys.modules["aiohttp.web"] = web_mod

    return routes


class LiveRouteRegistrationTests(unittest.TestCase):

    def test_blob_bus_registers_routes(self):
        routes = _install_fake_comfyui_runtime()
        # Reset module-level state.
        if "blob_bus" in sys.modules:
            del sys.modules["blob_bus"]
        import blob_bus
        # install() does the dir creation + reaper start + route register.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ok = blob_bus.install(comfyui_output_dir=td)
            self.assertTrue(ok, "blob_bus.install() should succeed")
            self.assertTrue(blob_bus.is_available())
        self.assertIn(("POST", "/spellcaster/blob/put"), routes)
        self.assertIn(("GET", "/spellcaster/blob/list"), routes)
        self.assertIn(("GET", "/spellcaster/blob/{hash}"), routes)

    def test_pack_version_route_registers_and_serves_manifest(self):
        routes = _install_fake_comfyui_runtime()
        if "pack_version_route" in sys.modules:
            del sys.modules["pack_version_route"]
        import pack_version_route
        ok = pack_version_route.is_available()
        self.assertTrue(ok)
        self.assertIn(("GET", "/spellcaster/version"), routes)

        # Invoke the handler with a fake request.
        handler = routes[("GET", "/spellcaster/version")]
        class FakeReq: pass
        resp = asyncio.run(handler(FakeReq()))
        self.assertEqual(resp.status, 200)
        manifest = resp.json_payload
        self.assertEqual(manifest["pack"], "ComfyUI-Spellcaster")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertIn("subsystems", manifest)
        self.assertIn("security", manifest)
        self.assertIn("routes", manifest)
        # The handshake must mention the new endpoint as a known route.
        self.assertIn("/spellcaster/version", manifest["routes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
