"""Pure-Python ChaCha20-Poly1305 AEAD — RFC 7539/8439.

Fallback for environments where `cryptography` isn't installed
(e.g., GIMP 3's MinGW-built embedded Python 3.12, which ships with
no AEAD primitives and no pip). Uses only stdlib (struct, hmac,
hashlib).

~10-30× slower than the cryptography package's C impl — acceptable
for wizard sentinel writes + occasional envelope ops; callers that
encrypt MB of data per second should install the real cryptography
package when possible.

Security note: constant-time MAC compare via hmac.compare_digest.
No constant-time guarantees on the cipher itself (pure Python —
timing side channels are possible). For the threat model where
the machine is already trusted (offline workstation), this is
acceptable. For transport over hostile networks, use the real
cryptography package.
"""
from __future__ import annotations

import hmac
import struct


# ── ChaCha20 ──────────────────────────────────────────────────────

_SIGMA = b"expand 32-byte k"


def _rotl32(v: int, c: int) -> int:
    return ((v << c) & 0xFFFFFFFF) | (v >> (32 - c))


def _quarter_round(s: list[int], a: int, b: int, c: int, d: int) -> None:
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF; s[d] = _rotl32(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF; s[b] = _rotl32(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & 0xFFFFFFFF; s[d] = _rotl32(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & 0xFFFFFFFF; s[b] = _rotl32(s[b] ^ s[c], 7)


def _chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    assert len(key) == 32 and len(nonce) == 12
    state = list(struct.unpack("<4I", _SIGMA))
    state += list(struct.unpack("<8I", key))
    state += [counter]
    state += list(struct.unpack("<3I", nonce))
    working = state.copy()
    for _ in range(10):
        _quarter_round(working, 0, 4, 8, 12)
        _quarter_round(working, 1, 5, 9, 13)
        _quarter_round(working, 2, 6, 10, 14)
        _quarter_round(working, 3, 7, 11, 15)
        _quarter_round(working, 0, 5, 10, 15)
        _quarter_round(working, 1, 6, 11, 12)
        _quarter_round(working, 2, 7, 8, 13)
        _quarter_round(working, 3, 4, 9, 14)
    out = [(working[i] + state[i]) & 0xFFFFFFFF for i in range(16)]
    return struct.pack("<16I", *out)


def _chacha20_xor(key: bytes, counter: int, nonce: bytes,
                    data: bytes) -> bytes:
    out = bytearray(len(data))
    block_counter = counter
    i = 0
    while i < len(data):
        block = _chacha20_block(key, block_counter, nonce)
        chunk = data[i:i + 64]
        for j, b in enumerate(chunk):
            out[i + j] = b ^ block[j]
        i += 64
        block_counter += 1
    return bytes(out)


# ── Poly1305 ──────────────────────────────────────────────────────

_P1305 = (1 << 130) - 5


def _poly1305_mac(msg: bytes, key: bytes) -> bytes:
    assert len(key) == 32
    r = int.from_bytes(key[:16], "little")
    # Clamp r per RFC 7539 §2.5.1
    r &= 0x0ffffffc0ffffffc0ffffffc0fffffff
    s = int.from_bytes(key[16:], "little")
    acc = 0
    i = 0
    while i < len(msg):
        chunk = msg[i:i + 16]
        n = int.from_bytes(chunk + b"\x01", "little")
        acc = (acc + n) % _P1305
        acc = (acc * r) % _P1305
        i += 16
    acc = (acc + s) & ((1 << 128) - 1)
    return acc.to_bytes(16, "little")


# ── ChaCha20-Poly1305 AEAD ────────────────────────────────────────

def _pad16(x: bytes) -> bytes:
    if len(x) % 16 == 0:
        return b""
    return b"\x00" * (16 - (len(x) % 16))


def _poly_key(key: bytes, nonce: bytes) -> bytes:
    return _chacha20_block(key, 0, nonce)[:32]


class ChaCha20Poly1305:
    """Drop-in replacement for the cryptography package's class of
    the same name — the surface is limited to encrypt(nonce, data,
    aad) / decrypt(nonce, data, aad) so at_rest.py's calls work
    unchanged."""

    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("ChaCha20-Poly1305 key must be 32 bytes")
        self._key = bytes(key)

    def encrypt(self, nonce: bytes, data: bytes,
                associated_data: bytes | None) -> bytes:
        if len(nonce) != 12:
            raise ValueError("nonce must be 12 bytes")
        aad = associated_data or b""
        poly_key = _poly_key(self._key, nonce)
        ct = _chacha20_xor(self._key, 1, nonce, data)
        mac_data = (
            aad + _pad16(aad)
            + ct + _pad16(ct)
            + struct.pack("<QQ", len(aad), len(ct))
        )
        tag = _poly1305_mac(mac_data, poly_key)
        return ct + tag

    def decrypt(self, nonce: bytes, data: bytes,
                associated_data: bytes | None) -> bytes:
        if len(nonce) != 12:
            raise ValueError("nonce must be 12 bytes")
        if len(data) < 16:
            raise ValueError("ciphertext too short (no tag)")
        aad = associated_data or b""
        ct = data[:-16]
        expected_tag = data[-16:]
        poly_key = _poly_key(self._key, nonce)
        mac_data = (
            aad + _pad16(aad)
            + ct + _pad16(ct)
            + struct.pack("<QQ", len(aad), len(ct))
        )
        computed_tag = _poly1305_mac(mac_data, poly_key)
        if not hmac.compare_digest(expected_tag, computed_tag):
            raise ValueError("AEAD tag mismatch — ciphertext tampered")
        return _chacha20_xor(self._key, 1, nonce, ct)


# ── HKDF-SHA256 (RFC 5869) — pure stdlib ──────────────────────────

import hashlib


def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    if not salt:
        salt = b"\x00" * hashlib.sha256().digest_size
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    out = b""
    t = b""
    counter = 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        out += t
        counter += 1
    return out[:length]


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """HKDF-SHA256 — RFC 5869 single-step extract+expand."""
    prk = _hkdf_extract(salt, ikm)
    return _hkdf_expand(prk, info, length)


# ── Self-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    key = os.urandom(32)
    nonce = os.urandom(12)
    for n in (0, 1, 63, 64, 65, 1024, 65536):
        body = os.urandom(n)
        aead = ChaCha20Poly1305(key)
        ct = aead.encrypt(nonce, body, b"header")
        pt = aead.decrypt(nonce, ct, b"header")
        assert pt == body, f"round-trip failed for n={n}"
    # Tamper detection
    ct = ChaCha20Poly1305(key).encrypt(nonce, b"hello", None)
    bad = bytearray(ct); bad[-1] ^= 1
    try:
        ChaCha20Poly1305(key).decrypt(nonce, bytes(bad), None)
        print("FAIL: tamper not detected")
    except ValueError:
        print("OK tamper detected")
    # HKDF RFC 5869 test vector (Case 1)
    ikm = bytes.fromhex("0b" * 22)
    salt = bytes.fromhex("000102030405060708090a0b0c")
    info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
    expected = bytes.fromhex("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865")
    got = hkdf_sha256(ikm, salt, info, 42)
    assert got == expected, f"HKDF test vector mismatch"
    print("OK all round-trips + HKDF vector 1")
