"""Unit tests for the blob_bus security enhancements.

Covers:
  * Plain put/get round-trip (regression — make sure the new params
    didn't break the existing flow)
  * One-shot semantics: GET consumes; second GET 404s
  * Encrypted put/get round-trip when a stub key is wired
  * Encrypt-without-token returns the explicit error (fail-closed)
  * Localhost-only mode flag exposed via the env var
  * Listing reflects the new flags

Doesn't exercise the HTTP layer — that needs a live ComfyUI. Direct
calls into the module's public API give us deterministic, fast
coverage and the HTTP shim is a thin parser around these functions.

Run:  python -m pytest tests/test_blob_bus_security.py -v
Or:   python tests/test_blob_bus_security.py
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Pack root on sys.path so `blob_bus` (sitting at pack root) is
# importable as a top-level module without dragging in the ComfyUI
# import-time side effects from __init__.py.
_HERE = Path(__file__).resolve().parent
_PACK_DIR = _HERE.parent
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))


class BlobBusSmokeTests(unittest.TestCase):
    """Public API contract — round-trip + the new flags."""

    def setUp(self):
        # Fresh state per test so a one-shot consumed in one test doesn't
        # leak into the next. The module-level dicts get reset.
        if "blob_bus" in sys.modules:
            importlib.reload(sys.modules["blob_bus"])
        import blob_bus
        self.bb = blob_bus
        self.tmp = tempfile.TemporaryDirectory()
        # install() reads env vars at module-scope — re-set then re-import.
        self.bb._store_dir = self.tmp.name
        self.bb._available = True
        # Fresh dict
        self.bb._blobs.clear()
        self.bb._total_bytes = 0
        # Clear the cached bus key so each test starts with a clean
        # encryption-availability slate.
        self.bb._bus_key_cache = None

    def tearDown(self):
        self.tmp.cleanup()

    # ── plain round-trip (regression) ───────────────────────────────

    def test_put_then_get_returns_same_bytes(self):
        payload = b"hello-world" * 100
        rec = self.bb.put(payload, kind="generation", origin="test")
        self.assertNotIn("error", rec)
        self.assertEqual(rec["size"], len(payload))
        self.assertFalse(rec["encrypted"])
        self.assertFalse(rec["one_shot"])
        got = self.bb.get(rec["hash"])
        self.assertIsNotNone(got)
        data, mime = got
        self.assertEqual(data, payload)
        # Plain bytes — mime sniff falls back to octet-stream.
        self.assertEqual(mime, "application/octet-stream")

    def test_put_then_get_png_sniffs_mime(self):
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
        rec = self.bb.put(png_header)
        self.assertEqual(rec["mime"], "image/png")
        _, mime = self.bb.get(rec["hash"])
        self.assertEqual(mime, "image/png")

    # ── one-shot semantics ──────────────────────────────────────────

    def test_one_shot_consumed_after_first_get(self):
        rec = self.bb.put(b"sensitive-mask", one_shot=True)
        self.assertTrue(rec["one_shot"])
        h = rec["hash"]
        first = self.bb.get(h)
        self.assertIsNotNone(first, "first GET should succeed")
        self.assertEqual(first[0], b"sensitive-mask")
        # Second GET — blob is gone.
        self.assertIsNone(self.bb.get(h),
                          "second GET must 404 (one_shot consumed)")

    def test_one_shot_persists_via_dedup(self):
        # If a non-one-shot upload is dedup'd against a one-shot record,
        # the flag stays sticky (conservative: prefer cleanup).
        rec1 = self.bb.put(b"shared-bytes", one_shot=True)
        rec2 = self.bb.put(b"shared-bytes", one_shot=False)
        self.assertEqual(rec1["hash"], rec2["hash"])
        # The dedup'd record should still report one_shot=True.
        # First GET consumes; second 404s.
        self.assertIsNotNone(self.bb.get(rec1["hash"]))
        self.assertIsNone(self.bb.get(rec1["hash"]))

    # ── encryption: fail-closed when no token ───────────────────────

    def test_encrypt_without_token_fails_closed(self):
        # No auth token in this test env — encrypt=True must reject.
        rec = self.bb.put(b"would-be-plaintext", encrypt=True)
        self.assertIn("error", rec)
        self.assertIn("auth token", rec["error"])
        # And nothing should have been written.
        self.assertEqual(len(self.bb._blobs), 0)

    # ── encryption: with stubbed key ────────────────────────────────

    def test_encrypt_round_trip_with_stub_key(self):
        # Stub the key resolution + wire_envelope so we don't depend
        # on the private_pipeline being importable in this env.
        captured: dict = {}

        def fake_resolve_bus_key():
            return b"\x42" * 32  # 32-byte ChaCha20 key

        def fake_wrap(plain, key, kind=0x03):
            captured["wrapped"] = plain
            return b"V1W\n" + b"\x03\x00\x00\x00" + b"\x00" * 12 + plain

        def fake_unwrap_bytes(envelope):
            # Mirror the wrap layout: header is 20 bytes, body follows.
            if envelope[:4] != b"V1W\n":
                return None
            return envelope[20:]

        self.bb._resolve_bus_key = fake_resolve_bus_key
        self.bb._wrap_bytes = lambda data: fake_wrap(data, b"x")
        self.bb._unwrap_bytes = fake_unwrap_bytes

        plaintext = b"hello-encrypted-world"
        rec = self.bb.put(plaintext, encrypt=True)
        self.assertNotIn("error", rec, msg=str(rec))
        self.assertTrue(rec["encrypted"])
        self.assertEqual(rec["plain_mime"], "application/octet-stream")
        # On-disk size is ciphertext (wrapper + 20-byte header).
        self.assertEqual(rec["size"], len(plaintext) + 20)
        # GET with decrypt=True (default) returns plaintext.
        data, _mime = self.bb.get(rec["hash"])
        self.assertEqual(data, plaintext)
        # GET with decrypt=False returns the ciphertext envelope.
        data_raw, mime_raw = self.bb.get(rec["hash"], decrypt=False)
        self.assertTrue(data_raw.startswith(b"V1W\n"))
        self.assertEqual(mime_raw, "application/x-spellcaster-v1w")

    def test_encrypt_with_key_but_get_without_key_returns_none(self):
        # Wrote with a key; later the key is unavailable → decrypt=True
        # must 404 rather than leak ciphertext.
        self.bb._resolve_bus_key = lambda: b"\x55" * 32
        self.bb._wrap_bytes = lambda plain: b"V1W\n" + b"\x03\x00\x00\x00" + b"\x00" * 12 + plain
        rec = self.bb.put(b"secret", encrypt=True)
        self.assertTrue(rec["encrypted"])
        # Now stub _unwrap_bytes to fail (simulating key rotation /
        # token removal between put and get).
        self.bb._unwrap_bytes = lambda env: None
        self.assertIsNone(self.bb.get(rec["hash"]),
                          "GET with decrypt=True must 404 when key is gone")
        # raw=False (default) means decrypt=True; explicit decrypt=False
        # still lets the caller pull the ciphertext envelope.
        raw = self.bb.get(rec["hash"], decrypt=False)
        self.assertIsNotNone(raw)

    # ── listing reflects flags ──────────────────────────────────────

    def test_list_blobs_reports_new_flags(self):
        self.bb.put(b"plain", origin="test")
        self.bb.put(b"oneshot", origin="test", one_shot=True)
        listing = self.bb.list_blobs()
        flags = {(b["origin"], b["encrypted"], b["one_shot"])
                 for b in listing["blobs"]}
        self.assertIn(("test", False, False), flags)
        self.assertIn(("test", False, True), flags)
        # Posture fields surfaced too.
        self.assertIn("localhost_only", listing)
        self.assertIn("encryption_available", listing)


class LocalhostOnlyFlagTests(unittest.TestCase):
    """The env var is read at module import — exercising via reload."""

    def test_env_var_truthy_parsed_correctly(self):
        # Re-import with the env var set; verify the module-level flag.
        os.environ["SPELLCASTER_LOCALHOST_ONLY"] = "1"
        try:
            if "blob_bus" in sys.modules:
                importlib.reload(sys.modules["blob_bus"])
            import blob_bus
            self.assertTrue(blob_bus.LOCALHOST_ONLY)
        finally:
            del os.environ["SPELLCASTER_LOCALHOST_ONLY"]
            # Reload again so subsequent tests start clean.
            if "blob_bus" in sys.modules:
                importlib.reload(sys.modules["blob_bus"])

    def test_is_loopback_classifier(self):
        if "blob_bus" in sys.modules:
            importlib.reload(sys.modules["blob_bus"])
        import blob_bus
        self.assertTrue(blob_bus._is_loopback("127.0.0.1"))
        self.assertTrue(blob_bus._is_loopback("::1"))
        self.assertTrue(blob_bus._is_loopback("localhost"))
        self.assertFalse(blob_bus._is_loopback("192.168.1.42"))
        self.assertFalse(blob_bus._is_loopback("8.8.8.8"))
        self.assertFalse(blob_bus._is_loopback(""))
        self.assertFalse(blob_bus._is_loopback("evil.example.com"))


class PackVersionManifestTests(unittest.TestCase):
    """Smoke test for the unified version endpoint composer."""

    def test_manifest_shape(self):
        # pack_version_route does best-effort imports — when run from
        # the pack dir as a top-level module, the relative imports
        # will fail and the subsystem flags fall back to False. That's
        # the documented "no ComfyUI runtime" path.
        sys.path.insert(0, str(_PACK_DIR))
        if "pack_version_route" in sys.modules:
            importlib.reload(sys.modules["pack_version_route"])
        # Import via the package path so the relative imports inside
        # _build_manifest can resolve. If the package import fails
        # (no comfyui-spellcaster on sys.path), fall back to direct.
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "pack_version_route",
                _PACK_DIR / "pack_version_route.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:
            self.fail(f"could not load pack_version_route: {exc}")
        manifest = mod._build_manifest()
        self.assertEqual(manifest["pack"], "ComfyUI-Spellcaster")
        self.assertIn("pack_version", manifest)
        self.assertEqual(manifest["schema_version"], 1)
        # Subsystems is a dict with the expected key set.
        self.assertEqual(
            set(manifest["subsystems"].keys()),
            {"presence", "blob_bus", "privacy_cleanup",
             "model_repair", "private_crypto"},
        )
        # Security dict — at minimum the new flags exist.
        self.assertIn("localhost_only", manifest["security"])
        self.assertIn("blob_at_rest_aead", manifest["security"])
        self.assertIn("blob_one_shot", manifest["security"])
        # blob_one_shot is unconditional (the code path always exists).
        self.assertTrue(manifest["security"]["blob_one_shot"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
