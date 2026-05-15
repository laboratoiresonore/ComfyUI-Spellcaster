"""Spellcaster blob bus — Guild-less asset transport via ComfyUI.

Why this module exists
----------------------

Presence (presence.py) makes plugins aware of each other without the
Wizard Guild. But asset TRANSPORT still routes through the Guild's
``/api/assets/<hash>`` store: if the Guild is down, peers see each
other but can't hand bytes over. On a LAN with multiple machines,
Guild-on-one-host is a single point of failure for every workflow
that moves an image from one plugin to another.

Every plugin already talks to ComfyUI as a hard dependency. So the
obvious low-failure-rate blob store is ComfyUI itself. This module
adds three HTTP routes on ComfyUI's PromptServer that any plugin can
POST bytes to and any OTHER plugin can GET bytes from — the Guild is
cut out of the transport path entirely.

Endpoints
---------

    POST /spellcaster/blob/put
        Multipart form:
          - file: the raw bytes (required)
          - kind: short string label (optional, default "generation")
          - origin: publisher key ("gimp", "sillytavern", ...)
          - ttl_s: int, auto-expire after this many seconds (default 3600)
          - encrypt: "1" to wrap bytes in a V1W envelope on disk (private
                     add-on auth token must be configured; otherwise
                     400). Hash is computed over the CIPHERTEXT so the
                     content-addressed URL doesn't leak plaintext sha256.
          - one_shot: "1" to delete the blob after the first successful
                     GET (concurrent GETs all see the same bytes; later
                     GETs 404). Useful for sensitive workflow handoffs.
        Returns JSON:
          {hash, url, size, kind, origin, expires_at,
           encrypted: bool, one_shot: bool}
        The url is absolute (includes the scheme+host ComfyUI was
        reached on, inferred from the Host: header) so recipients on
        other LAN machines can fetch without knowing ComfyUI's address.

    GET /spellcaster/blob/<hash>
        Returns the raw bytes with the original content-type (inferred
        from the first bytes via `imghdr`/`mimetypes`). 404 if the
        hash is unknown or already expired. Encrypted blobs are
        transparently unwrapped before send when the server's auth
        token is configured.

    GET /spellcaster/blob/list
        Returns {blobs: [{hash, size, kind, origin, age_s, expires_in_s,
                          encrypted, one_shot}]}
        — the full catalog of still-live blobs, for UIs that want to
        show "recent cross-app assets".

TTL + eviction
--------------

Blobs live on disk under ``<comfyui-root>/output/spellcaster_bus/``
(same volume ComfyUI already uses, so size caps hit the same quota).
Every blob records ``expires_at``; a background reaper thread evicts
expired ones every 60 s. The default 1-hour TTL is a middle ground:
long enough for a user to finish a multi-plugin workflow
(DT→GIMP→Resolve), short enough that a forgotten blob doesn't pin
disk forever. Callers that need longer persistence should still push
through the Guild's AssetGallery.

Safety
------

* Hard ceiling of 256 MB per blob and 2 GB aggregate store size (see
  MAX_BLOB_BYTES / MAX_STORE_BYTES). Uploads exceeding either are
  rejected with 413.
* Incoming blobs are stored content-hash addressed (sha256); repeated
  uploads of the same bytes deduplicate on disk.
* Origin + kind are size-capped strings matching the rest of the
  Spellcaster presence charset — no arbitrary keys.
* No authentication on the put/get path itself — if your ComfyUI is
  exposed to hostile networks you have much worse problems; this
  module follows the same trust model as the presence broker (LAN +
  trusted clients). For tighter postures, set
  ``SPELLCASTER_LOCALHOST_ONLY=1`` to refuse non-loopback uploads, or
  pass ``encrypt=1`` to bind the bytes to the server's private auth
  token so an attacker who can read the on-disk file still gets only
  ciphertext.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import threading
import time
from typing import Any, Optional


# ── configuration ────────────────────────────────────────────────────

DEFAULT_TTL_S: float = 3600.0         # 1 hour
MAX_BLOB_BYTES: int = 256 * 1024 * 1024   # per-blob
MAX_STORE_BYTES: int = 2 * 1024 * 1024 * 1024  # aggregate
REAPER_INTERVAL_S: float = 60.0

MAX_KIND_LEN: int = 48
MAX_ORIGIN_LEN: int = 48

# Localhost-only mode: when the env var is "1" / "true" / "yes" at
# module import time, the put route refuses any client whose remote
# address isn't a loopback. The list/get routes stay open because the
# blobs themselves are protected (encrypted blobs need the auth token;
# unencrypted blobs were uploaded by trusted clients to begin with).
# Read at module import time only — flipping the env var at runtime
# requires a ComfyUI restart, which matches the rest of Spellcaster's
# env-var posture and avoids a TOCTOU window between probe + put.
def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


LOCALHOST_ONLY: bool = _env_truthy("SPELLCASTER_LOCALHOST_ONLY")


# ── state ────────────────────────────────────────────────────────────

_lock = threading.Lock()
_blobs: dict[str, dict] = {}
_total_bytes: int = 0
_reaper_started: bool = False
_available: bool = False

#: Where blobs live on disk — filled in by ``install()``.
_store_dir: Optional[str] = None


# ── at-rest encryption helpers ───────────────────────────────────────
#
# Bound to the same auth token used by private_crypto_nodes.py. When
# the token is unset the encryption path stays dormant (encrypt=1
# uploads get a clean 400 explaining the setup step). Key derivation
# uses a bus-specific HKDF info string so a compromise of the at-rest
# bus key doesn't reveal the wire / save-image keys (key separation
# per CLAUDE.md). Imports are deferred so a missing private_pipeline
# directory doesn't break the bus's plain-bytes path.


def _resolve_bus_key() -> Optional[bytes]:
    """Derive a bus-scoped key from the private add-on's auth token.

    Returns None if the token isn't configured on this server, OR if
    the wire_envelope module can't be imported (private add-on not
    installed in this pack). Cached: a single key derivation per
    ComfyUI process is enough; the token doesn't rotate at runtime.
    """
    global _bus_key_cache
    if _bus_key_cache is not None:
        return _bus_key_cache
    try:
        # Same lookup order as private_crypto_nodes._resolve_auth_token.
        from .private_crypto_nodes import _resolve_auth_token
        from .private_pipeline.wire_envelope import derive_key
    except Exception:
        return None
    try:
        token = _resolve_auth_token()
    except Exception:
        return None
    try:
        # Distinct HKDF info from the wire / save envelope so a
        # cryptanalytic break of one context doesn't pivot to the others.
        _bus_key_cache = derive_key(token, salt=b"spellcaster-blob-bus-v1")
        return _bus_key_cache
    except Exception:
        return None


_bus_key_cache: Optional[bytes] = None


def _wrap_bytes(plain: bytes) -> Optional[bytes]:
    """V1W-envelope-wrap plaintext. Returns None when the auth token
    or wire_envelope module is unavailable; callers must treat that
    as a hard 400 (we don't silently fall back to plaintext on disk
    when encrypt=1 was requested)."""
    key = _resolve_bus_key()
    if key is None:
        return None
    try:
        from .private_pipeline.wire_envelope import wrap, KIND_RAW
        return wrap(plain, key, kind=KIND_RAW)
    except Exception:
        return None


def _unwrap_bytes(envelope: bytes) -> Optional[bytes]:
    """Reverse of _wrap_bytes. Returns None on any failure so the
    GET handler can decide whether to 404 (no key, can't decrypt) or
    serve as-is."""
    key = _resolve_bus_key()
    if key is None:
        return None
    try:
        from .private_pipeline.wire_envelope import unwrap, is_wrapped
        if not is_wrapped(envelope):
            return None
        _kind, plain = unwrap(envelope, key)
        return plain
    except Exception:
        return None


def _is_loopback(addr: str) -> bool:
    """True iff ``addr`` is a loopback IP (IPv4 or IPv6). Defensively
    handles ipv6 zone ids and bracketed forms."""
    if not addr:
        return False
    # Strip ipv6 zone id (e.g. "fe80::1%eth0") + brackets.
    clean = addr.split("%", 1)[0].strip("[]")
    try:
        return ipaddress.ip_address(clean).is_loopback
    except ValueError:
        # Hostnames like "localhost" — treat the literal as loopback so
        # aiohttp setups that resolve client addr to a hostname don't
        # spuriously reject. Anything else is non-loopback.
        return clean.lower() in ("localhost",)


# ── helpers ──────────────────────────────────────────────────────────

def _safe_short(s: Any, max_len: int) -> str:
    if not isinstance(s, str):
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "0123456789-_")
    out = "".join(c for c in s[:max_len] if c in allowed)
    return out


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sniff_mime(data: bytes) -> str:
    """Best-effort content-type guess. Returns application/octet-stream
    when nothing matches; fine for non-image blobs."""
    # PNG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # JPEG (SOI marker)
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    # WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    # GIF
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    # MP4 / MOV
    if data[4:8] == b"ftyp":
        return "video/mp4"
    return "application/octet-stream"


def _reap_expired() -> None:
    """Evict expired blobs. Caller must hold _lock."""
    global _total_bytes
    now = time.time()
    dead = [h for h, b in _blobs.items() if b["expires_at"] < now]
    for h in dead:
        b = _blobs.pop(h)
        _total_bytes -= b["size"]
        try:
            os.unlink(b["path"])
        except OSError:
            pass


def _reaper_loop() -> None:
    while True:
        try:
            time.sleep(REAPER_INTERVAL_S)
            with _lock:
                _reap_expired()
        except Exception:
            pass


def _evict_to_fit(incoming_bytes: int) -> bool:
    """Drop oldest-expiring blobs until adding incoming_bytes fits
    under MAX_STORE_BYTES. Returns True on success, False if even a
    full flush wouldn't fit (incoming_bytes > MAX_STORE_BYTES)."""
    global _total_bytes
    if incoming_bytes > MAX_STORE_BYTES:
        return False
    while _total_bytes + incoming_bytes > MAX_STORE_BYTES and _blobs:
        oldest = min(_blobs.items(), key=lambda kv: kv[1]["expires_at"])[0]
        b = _blobs.pop(oldest)
        _total_bytes -= b["size"]
        try:
            os.unlink(b["path"])
        except OSError:
            pass
    return _total_bytes + incoming_bytes <= MAX_STORE_BYTES


# ── public API (also callable without HTTP) ──────────────────────────

def put(data: bytes, kind: str = "generation", origin: str = "unknown",
        ttl_s: Optional[float] = None,
        encrypt: bool = False, one_shot: bool = False) -> dict:
    """Store bytes + return a record ready to serialize. Deduplicates
    by content hash; re-upload of identical bytes just bumps the TTL.

    When ``encrypt=True``, the plaintext is wrapped in a V1W envelope
    using the server's private auth-token-derived key BEFORE hashing
    + writing to disk. The on-disk file is therefore ciphertext; a
    GET against the bus transparently unwraps before sending. If the
    auth token isn't configured the call returns an explicit error
    instead of falling back to plaintext on disk (fail closed).

    ``one_shot=True`` marks the blob for delete-after-first-GET.
    Useful for sensitive workflow handoffs (e.g. mask uploads) where
    persistence past the consumer isn't wanted.
    """
    if not isinstance(data, (bytes, bytearray)) or not data:
        return {"error": "empty body"}
    if len(data) > MAX_BLOB_BYTES:
        return {"error": f"exceeds MAX_BLOB_BYTES ({MAX_BLOB_BYTES})"}
    if _store_dir is None:
        return {"error": "blob bus not installed"}
    kind = _safe_short(kind, MAX_KIND_LEN) or "generation"
    origin = _safe_short(origin, MAX_ORIGIN_LEN) or "unknown"
    try:
        ttl = float(ttl_s) if ttl_s is not None else DEFAULT_TTL_S
    except (TypeError, ValueError):
        ttl = DEFAULT_TTL_S
    ttl = max(60.0, min(ttl, 24 * 3600.0))  # [1 min .. 24 h]

    plain_view = bytes(data)
    plain_mime = _sniff_mime(plain_view[:64])

    if encrypt:
        wrapped = _wrap_bytes(plain_view)
        if wrapped is None:
            # Fail closed — the caller asked for encryption and we
            # cannot deliver. NEVER silently store plaintext when
            # encrypt=1 was set.
            return {"error": (
                "encrypt=1 requested but the private auth token is "
                "not configured on this server. POST your token to "
                "/spellcaster/private/setup first, or copy it to "
                "<pack_dir>/.auth_token.")}
        stored = wrapped
        # Mime for an encrypted blob is opaque — peers that re-GET
        # without decrypt rights see only ciphertext.
        stored_mime = "application/x-spellcaster-v1w"
    else:
        stored = plain_view
        stored_mime = plain_mime

    # Hash is over what we actually store. For encrypted blobs that's
    # the ciphertext, which means the URL doesn't leak a plaintext
    # sha256 (defence against rainbow-table-style probes on known
    # asset hashes).
    h = _hash_bytes(stored)
    now = time.time()
    expires_at = now + ttl

    with _lock:
        _reap_expired()
        if h in _blobs:
            # Dedup — refresh TTL but don't rewrite bytes. One-shot is
            # "sticky on": if either upload marked it one-shot, the
            # blob keeps that flag (conservative — prefer over-aggressive
            # cleanup for a workflow-handoff blob).
            _blobs[h]["expires_at"] = expires_at
            if one_shot:
                _blobs[h]["one_shot"] = True
            return {
                "hash": h, "size": _blobs[h]["size"],
                "kind": _blobs[h]["kind"], "origin": _blobs[h]["origin"],
                "mime": _blobs[h]["mime"],
                "created_at": _blobs[h]["created_at"],
                "expires_at": expires_at,
                "encrypted": _blobs[h].get("encrypted", False),
                "one_shot": _blobs[h].get("one_shot", False),
                "plain_mime": _blobs[h].get("plain_mime"),
            }
        # New blob — evict if needed and write to disk.
        if not _evict_to_fit(len(stored)):
            return {"error": "store full"}
        path = os.path.join(_store_dir, h)
        try:
            with open(path, "wb") as f:
                f.write(stored)
        except OSError as e:
            return {"error": f"write failed: {e}"}
        global _total_bytes
        rec = {
            "hash": h,
            "size": len(stored),
            "kind": kind,
            "origin": origin,
            "mime": stored_mime,
            "plain_mime": plain_mime if encrypt else None,
            "encrypted": bool(encrypt),
            "one_shot": bool(one_shot),
            "created_at": now,
            "expires_at": expires_at,
            "path": path,
        }
        _blobs[h] = rec
        _total_bytes += len(stored)

    return {
        "hash": h, "size": rec["size"],
        "kind": rec["kind"], "origin": rec["origin"],
        "mime": rec["mime"],
        "plain_mime": rec["plain_mime"],
        "encrypted": rec["encrypted"],
        "one_shot": rec["one_shot"],
        "created_at": rec["created_at"],
        "expires_at": rec["expires_at"],
    }


def get(h: str, decrypt: bool = True) -> Optional[tuple[bytes, str]]:
    """Return (bytes, mime) for a hash, or None if expired/unknown.
    Bytes are read off disk on every call — the in-memory dict only
    holds metadata.

    When the stored blob is encrypted, the default ``decrypt=True``
    transparently unwraps using the server's auth-token-derived key
    and returns the plaintext + the sniffed original mime type. Pass
    ``decrypt=False`` to fetch the raw ciphertext (for peers that
    intend to forward the envelope to another consumer who holds the
    same key).

    Encrypted blobs with ``decrypt=True`` but no available key return
    None (404) rather than leaking ciphertext — opt-out is explicit.

    Side effect: if the blob was marked one_shot, this call consumes
    it. Subsequent gets return None. Concurrent gets that race past
    the lock all see the same bytes because the lock is held across
    the disk read + the dict mutation.
    """
    with _lock:
        _reap_expired()
        rec = _blobs.get(h)
        if rec is None:
            return None
        try:
            with open(rec["path"], "rb") as f:
                data = f.read()
        except OSError:
            return None

        is_enc = rec.get("encrypted", False)
        if is_enc and decrypt:
            unwrapped = _unwrap_bytes(data)
            if unwrapped is None:
                # Have ciphertext, can't decrypt — refuse rather than
                # silently emit V1W envelope as the public payload.
                return None
            payload = unwrapped
            payload_mime = rec.get("plain_mime") or _sniff_mime(payload[:64])
        else:
            payload = data
            payload_mime = rec["mime"]

        if rec.get("one_shot"):
            # Consume — drop from the catalog and unlink from disk.
            # We're still inside _lock so a concurrent get() that beat
            # us into the dict either sees the record (and gets a
            # consistent payload from this read) or doesn't.
            _blobs.pop(h, None)
            global _total_bytes
            _total_bytes -= rec["size"]
            try:
                os.unlink(rec["path"])
            except OSError:
                pass
        return payload, payload_mime


def list_blobs() -> dict:
    """Public catalog of live blobs — for UI that wants to show
    recent cross-app assets. Never includes the raw path."""
    now = time.time()
    with _lock:
        _reap_expired()
        out = []
        for rec in _blobs.values():
            out.append({
                "hash": rec["hash"],
                "size": rec["size"],
                "kind": rec["kind"],
                "origin": rec["origin"],
                "mime": rec["mime"],
                "encrypted": rec.get("encrypted", False),
                "one_shot": rec.get("one_shot", False),
                "age_s": round(now - rec["created_at"], 2),
                "expires_in_s": round(rec["expires_at"] - now, 2),
            })
        total = _total_bytes
    out.sort(key=lambda r: r["age_s"])
    return {
        "blobs": out,
        "total_bytes": total,
        "max_store_bytes": MAX_STORE_BYTES,
        "max_blob_bytes": MAX_BLOB_BYTES,
        "localhost_only": LOCALHOST_ONLY,
        "encryption_available": _resolve_bus_key() is not None,
    }


# ── HTTP route registration ──────────────────────────────────────────

def _build_url(request, h: str) -> str:
    """Assemble an absolute URL the requester can hand out to peers on
    the LAN. Uses the Host header ComfyUI saw — matches what the
    client typed. Falls back to a relative path if Host is missing."""
    try:
        host = request.headers.get("Host", "")
        scheme = request.scheme or "http"
        if host:
            return f"{scheme}://{host}/spellcaster/blob/{h}"
    except Exception:
        pass
    return f"/spellcaster/blob/{h}"


def _register_routes() -> bool:
    """Wire put/get/list onto ComfyUI's PromptServer. Returns True if
    the routes were attached."""
    try:
        from server import PromptServer  # ComfyUI singleton
    except Exception:  # pragma: no cover
        return False
    try:
        from aiohttp import web
    except Exception:  # pragma: no cover
        return False

    instance = getattr(PromptServer, "instance", None)
    if instance is None:
        return False
    routes = getattr(instance, "routes", None)
    if routes is None:
        return False

    def _truthy(s: Any) -> bool:
        if isinstance(s, bool):
            return s
        if not isinstance(s, str):
            return False
        return s.strip().lower() in ("1", "true", "yes", "on")

    @routes.post("/spellcaster/blob/put")
    async def _put(request):
        # LAN-isolation gate — when SPELLCASTER_LOCALHOST_ONLY is set,
        # refuse anything not from a loopback peer. The list/get paths
        # stay open because their payloads are either trust-uploaded
        # plaintext OR ciphertext that requires the auth token anyway.
        if LOCALHOST_ONLY:
            remote = getattr(request, "remote", "") or ""
            if not _is_loopback(remote):
                return web.json_response(
                    {"error": ("blob bus is in localhost-only mode "
                               "(SPELLCASTER_LOCALHOST_ONLY=1); refusing "
                               f"non-loopback client {remote!r}")},
                    status=403)
        # Accept multipart OR raw body. Multipart lets us carry
        # kind/origin/ttl_s/encrypt/one_shot in the same request; raw
        # is the curl-friendly shortcut.
        kind = "generation"
        origin = "unknown"
        ttl_s: Optional[float] = None
        encrypt = False
        one_shot = False
        data: bytes = b""
        try:
            if request.content_type and "multipart" in request.content_type:
                reader = await request.multipart()
                while True:
                    part = await reader.next()
                    if part is None:
                        break
                    if part.name == "file":
                        data = await part.read(decode=False)
                    elif part.name == "kind":
                        kind = (await part.text()).strip()
                    elif part.name == "origin":
                        origin = (await part.text()).strip()
                    elif part.name == "ttl_s":
                        try:
                            ttl_s = float((await part.text()).strip())
                        except ValueError:
                            pass
                    elif part.name == "encrypt":
                        encrypt = _truthy(await part.text())
                    elif part.name == "one_shot":
                        one_shot = _truthy(await part.text())
            else:
                data = await request.read()
                kind = request.query.get("kind", kind)
                origin = request.query.get("origin", origin)
                try:
                    ttl_s = float(request.query.get("ttl_s") or "") or None
                except ValueError:
                    ttl_s = None
                encrypt = _truthy(request.query.get("encrypt"))
                one_shot = _truthy(request.query.get("one_shot"))
        except Exception as e:
            return web.json_response({"error": f"read failed: {e}"}, status=400)

        result = put(data, kind=kind, origin=origin, ttl_s=ttl_s,
                     encrypt=encrypt, one_shot=one_shot)
        if "error" in result:
            err = result["error"]
            if "exceeds" in err:
                status = 413
            elif "encrypt=1 requested" in err:
                # 503 — capability unavailable on this server. Caller
                # should re-issue without encrypt=1 OR provision the
                # auth token first.
                status = 503
            else:
                status = 400
            return web.json_response(result, status=status)
        result["url"] = _build_url(request, result["hash"])
        return web.json_response(result)

    @routes.get("/spellcaster/blob/list")
    async def _list(_request):
        return web.json_response(list_blobs())

    @routes.get("/spellcaster/blob/{hash}")
    async def _get(request):
        h = request.match_info.get("hash", "")
        if not h or len(h) != 64 or not all(c in "0123456789abcdef" for c in h):
            return web.json_response({"error": "bad hash"}, status=400)
        # ?raw=1 returns ciphertext for encrypted blobs (used when the
        # caller intends to forward the envelope to another peer who
        # has the auth token). Default decrypts transparently.
        decrypt = not _truthy(request.query.get("raw"))
        got = get(h, decrypt=decrypt)
        if got is None:
            return web.json_response({"error": "not found"}, status=404)
        data, mime = got
        return web.Response(body=data, content_type=mime)

    return True


def install(comfyui_output_dir: Optional[str] = None) -> bool:
    """Called from __init__.py at ComfyUI startup. Idempotent. Creates
    the blob dir, starts the reaper thread, and registers routes."""
    global _store_dir, _reaper_started, _available
    if _available:
        return True
    if comfyui_output_dir is None:
        # The pack lives at custom_nodes/<name>/; ComfyUI root is two
        # levels up; output/ is next to it.
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(here))
        comfyui_output_dir = os.path.join(root, "output", "spellcaster_bus")
    _store_dir = comfyui_output_dir
    try:
        os.makedirs(_store_dir, exist_ok=True)
    except OSError:
        _store_dir = None
        return False

    # Clean stale blobs from a prior process — they have no metadata
    # in-memory any more, so the reaper can't touch them. One-shot
    # sweep at startup keeps the dir from growing unboundedly across
    # ComfyUI restarts.
    try:
        for name in os.listdir(_store_dir):
            if len(name) == 64:  # sha256-hex shape
                try:
                    os.unlink(os.path.join(_store_dir, name))
                except OSError:
                    pass
    except OSError:
        pass

    if not _reaper_started:
        t = threading.Thread(target=_reaper_loop, daemon=True,
                              name="spellcaster-blob-reaper")
        t.start()
        _reaper_started = True

    _available = _register_routes()
    return _available


def is_available() -> bool:
    return _available
