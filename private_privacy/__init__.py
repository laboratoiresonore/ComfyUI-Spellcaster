"""Vendored the private downstream distribution crypto primitives — ChaCha20-Poly1305 wire
envelope (V1W magic). Self-contained: works whether the system has
the `cryptography` package or only stdlib.

Same on-the-wire format as the client-side private-pipeline_privacy/
in the GIMP plug-in. Different `info` strings prevent cross-context
key reuse (per CLAUDE.md key-separation rule)."""

from . import wire_envelope

__all__ = ["wire_envelope"]
