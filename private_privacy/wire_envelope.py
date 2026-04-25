"""the private downstream distribution end-to-end workflow envelope.

Used for both transport (plug-in -> server upload / server -> plug-in
download) AND at-rest (files sit on the server's disk encrypted;
never appear in plaintext outside the running workflow). Both sides
share a key derived from the auth_token.

Envelope layout:
    0..3    "V1W\\n"      magic  (distinguishes from at_rest "V1R\\n")
    4       kind        0x01 image / 0x02 video / 0x03 raw bytes
    5..7    reserved    zero
    8..19   nonce       ChaCha20-Poly1305 12-byte nonce
    20..end ciphertext  ChaCha20-Poly1305 ct + 16-byte tag

HKDF info = b"private-wire-v1" (different from transport_aead +
at_rest so a compromise of one doesn't reveal the others).

This module is shared:
  - plug-in side vendors it to wrap pre-upload + unwrap post-download
  - ComfyUI pack side uses it in PrivateDecryptLoadImage /
    PrivateEncryptSaveImage / video equivalents
"""
from __future__ import annotations

import os
import struct
from typing import Tuple, Union

from pathlib import Path as _Path
_USING_PURE_FALLBACK = False
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
except ImportError:
    # Same fallback as at_rest.py — see its block for the rationale.
    _USING_PURE_FALLBACK = True
    import importlib.util as _iu
    _PURE_PATH = _Path(__file__).parent / "_chacha20poly1305_pure.py"
    _spec = _iu.spec_from_file_location(
        "private_chacha20poly1305_pure_wire", str(_PURE_PATH))
    _pure = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(_pure)  # type: ignore[union-attr]
    ChaCha20Poly1305 = _pure.ChaCha20Poly1305
    _pure_hkdf = _pure.hkdf_sha256
    hashes = None  # type: ignore[assignment]
    class HKDF:  # type: ignore[no-redef]
        def __init__(self, *, algorithm, length, salt, info):
            # Pure-Python fallback only knows SHA-256. Reject anything
            # other than the explicit None sentinel — silent algo
            # downgrade is a security bug.
            if algorithm is not None:
                raise NotImplementedError(
                    "pure-Python HKDF fallback is SHA-256 only; "
                    f"got algorithm={algorithm!r}")
            self._length = length
            self._salt = salt or b""
            self._info = info
        def derive(self, ikm):
            return _pure_hkdf(ikm, self._salt, self._info, self._length)


MAGIC        = b"V1W\n"
MAGIC_LEN    = 4
HEADER_LEN   = 20       # magic(4) + kind(1) + reserved(3) + nonce(12)
NONCE_LEN    = 12
TAG_LEN      = 16
HKDF_INFO    = b"private-wire-v1"
HKDF_LEN     = 32

KIND_IMAGE = 0x01
KIND_VIDEO = 0x02
KIND_RAW   = 0x03


class WireError(RuntimeError):
    """Any envelope handling error. Caller must fail closed."""


def derive_key(auth_token: Union[str, bytes], salt: bytes = b"") -> bytes:
    """See at_rest.derive_key for the security contract — same rules
    apply (high-entropy IKM required, key separation via info string)."""
    if isinstance(auth_token, str):
        auth_token = auth_token.encode("utf-8")
    if len(auth_token) < 16:
        raise WireError(
            f"auth_token too short ({len(auth_token)} bytes); "
            f"need >=16 bytes of high-entropy material")
    return HKDF(
        algorithm=None if _USING_PURE_FALLBACK else hashes.SHA256(),
        length=HKDF_LEN,
        salt=salt,
        info=HKDF_INFO,
    ).derive(auth_token)


def wrap(plaintext: bytes, key: bytes, kind: int = KIND_RAW) -> bytes:
    """Build a full envelope.

    SECURITY: header (MAGIC + KIND + reserved bytes) is bound as AAD.
    An attacker who flips the KIND byte to make the receiver call
    expect_kind=KIND_VIDEO on image plaintext breaks the Poly1305 tag,
    so kind misinterpretation fails closed."""
    if len(key) != HKDF_LEN:
        raise WireError(f"key must be {HKDF_LEN} bytes")
    if kind not in (KIND_IMAGE, KIND_VIDEO, KIND_RAW):
        raise WireError(f"bad kind: {kind}")
    nonce = os.urandom(NONCE_LEN)
    header = MAGIC + struct.pack(">B3x", kind) + nonce  # 20 bytes
    try:
        ct = ChaCha20Poly1305(key).encrypt(nonce, plaintext, header)
    except Exception as exc:
        raise WireError(f"encrypt failed: {exc}") from exc
    return header + ct


def unwrap(envelope: bytes, key: bytes,
           expect_kind: int | None = None) -> Tuple[int, bytes]:
    """Return (kind, plaintext). Raises on tamper / bad magic /
    wrong kind if expect_kind is given. AAD-bound header MUST match
    the original encrypt-side header byte-for-byte (Poly1305 tag
    verifies this)."""
    if len(key) != HKDF_LEN:
        raise WireError(f"key must be {HKDF_LEN} bytes")
    if len(envelope) < HEADER_LEN + TAG_LEN:
        raise WireError("envelope too short")
    if envelope[:MAGIC_LEN] != MAGIC:
        raise WireError(
            f"bad magic {envelope[:MAGIC_LEN]!r} -- not a "
            "the private downstream distribution wire envelope")
    header = envelope[:HEADER_LEN]
    kind = envelope[4]
    nonce = envelope[8:20]
    ct = envelope[HEADER_LEN:]
    if expect_kind is not None and kind != expect_kind:
        raise WireError(f"kind mismatch: got {kind}, want {expect_kind}")
    try:
        # AAD must match what wrap bound — the full 20-byte header.
        # A KIND-byte flip here changes header, breaks Poly1305.
        plain = ChaCha20Poly1305(key).decrypt(nonce, ct, header)
    except Exception as exc:
        raise WireError(f"decrypt failed: {exc}") from exc
    return kind, plain


def is_wrapped(data: bytes) -> bool:
    """Quick sniff used by bidirectional handlers that have to
    cope with both legacy plaintext and new encrypted bodies
    during migration."""
    return len(data) >= MAGIC_LEN and data[:MAGIC_LEN] == MAGIC


# --- CLI smoke ----------------------------------------------------

if __name__ == "__main__":
    import io
    tok = b"private-pipeline-wire-dev-token-long-enough-for-selftest-pass"
    key = derive_key(tok)
    for kind in (KIND_IMAGE, KIND_VIDEO, KIND_RAW):
        body = os.urandom(4096)
        env = wrap(body, key, kind)
        k, back = unwrap(env, key, expect_kind=kind)
        assert k == kind
        assert back == body
        assert is_wrapped(env)
        assert not is_wrapped(body)
    # tamper
    env = wrap(b"tamper-me", key, KIND_RAW)
    bad = bytearray(env); bad[-1] ^= 1
    try:
        unwrap(bytes(bad), key)
        print("FAIL: tamper not detected"); raise SystemExit(1)
    except WireError:
        print("OK tamper detected")
    print("OK 3-kind round-trip")
