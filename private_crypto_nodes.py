"""Private add-on server-side crypto nodes.

Closes the on-server plaintext window: input bytes arrive AEAD-wrapped
in a V1W envelope, get decrypted in RAM, the workflow runs, and the
output is re-encrypted before SaveImage. Result: at no point does a
plaintext PNG sit on the server's disk.

Pair with the client-side `_ai_helper.py` workflow rewriter that
detects these nodes via `/spellcaster/private/version` and substitutes
them for `LoadImage` / `SaveImage` automatically.

Auth-token resolution:
  1. `~/.spellcaster/auth_token` on the SERVER's filesystem (preferred —
     the user runs a one-time setup that writes this file).
  2. `SPELLCASTER_PRIVATE_AUTH_TOKEN` env var (CI / containerized deploys).
  3. Fail with a clear error pointing to the setup step.

The token is the same one the client uses for derive_key(); we both
derive `wire_envelope.derive_key(token, info=b"private-wire-v1")` so
the keys match.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from .private_pipeline import wire_envelope


_TOKEN_CACHE: bytes | None = None


def _resolve_auth_token() -> bytes:
    """Find the auth token on the server. Cached after first read.
    Lookup order:
      1. ~/.spellcaster/auth_token (POSIX-canonical)
      2. SPELLCASTER_PRIVATE_AUTH_TOKEN env var
      3. <pack_dir>/.auth_token (deployed by Update_ComfyUI.exe — the
         normal path for cross-machine LAN setups where the user
         doesn't want to fiddle with the server's home dir)
    """
    global _TOKEN_CACHE
    if _TOKEN_CACHE is not None:
        return _TOKEN_CACHE

    # 1. Standard location.
    p = Path.home() / ".spellcaster" / "auth_token"
    if p.is_file():
        token = p.read_text(encoding="utf-8").strip().encode("utf-8")
        if token:
            _TOKEN_CACHE = token
            return token

    # 2. Env var.
    env_tok = os.environ.get("SPELLCASTER_PRIVATE_AUTH_TOKEN", "").strip()
    if env_tok:
        _TOKEN_CACHE = env_tok.encode("utf-8")
        return _TOKEN_CACHE

    # 3. Pack-local fallback — Update_ComfyUI.exe writes here when
    #    deploying to a remote ComfyUI install over SMB/UNC.
    pack_local = Path(__file__).resolve().parent / ".auth_token"
    if pack_local.is_file():
        token = pack_local.read_text(encoding="utf-8").strip().encode("utf-8")
        if token:
            _TOKEN_CACHE = token
            return token

    raise RuntimeError(
        f"Private add-on auth token not found.\n\n"
        f"Looked for (in order):\n"
        f"  1. {p}\n"
        f"  2. SPELLCASTER_PRIVATE_AUTH_TOKEN env var\n"
        f"  3. {pack_local}\n\n"
        f"Run Update_ComfyUI.exe on your CLIENT machine to deploy the\n"
        f"token automatically, OR copy your client-side\n"
        f"~/.spellcaster/auth_token to one of the locations above.")


def _pil_to_tensor(im: Image.Image) -> torch.Tensor:
    """ComfyUI image tensor convention: float32 [B, H, W, C] in 0..1."""
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    arr = np.array(im).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 4:  # drop alpha for std workflows; keep mask separately
        rgb = arr[..., :3]
        return torch.from_numpy(rgb)[None, ...]
    return torch.from_numpy(arr)[None, ...]


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Reverse: float32 [1, H, W, 3] → PIL RGB."""
    if t.ndim == 4:
        t = t[0]
    arr = (t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        return Image.fromarray(arr[..., 0], mode="L")
    return Image.fromarray(arr, mode="RGB")


# ─── PrivateDecryptLoadImage ────────────────────────────────────────

class PrivateDecryptLoadImage:
    """Replaces LoadImage when the client uploaded a wire-encrypted
    blob. The blob arrives as a base64-encoded `V1W` envelope embedded
    in the prompt JSON. We decrypt in RAM, decode the inner PNG, and
    return the image tensor. The plaintext NEVER touches disk on this
    server."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encrypted_b64": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "base64-encoded V1W envelope",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY    = "spellcaster/private/crypto"
    FUNCTION    = "load"

    def load(self, encrypted_b64: str):
        if not encrypted_b64:
            raise ValueError(
                "PrivateDecryptLoadImage: encrypted_b64 is empty. "
                "Did the client wrap the upload via wire_envelope.wrap()?")
        try:
            envelope = base64.b64decode(encrypted_b64)
        except Exception as exc:
            raise ValueError(f"bad base64: {exc}") from exc

        token = _resolve_auth_token()
        key = wire_envelope.derive_key(token)
        try:
            kind, png_bytes = wire_envelope.unwrap(
                envelope, key, expect_kind=wire_envelope.KIND_IMAGE)
        except Exception as exc:
            raise RuntimeError(
                f"PrivateDecryptLoadImage: AEAD unwrap failed — wrong "
                f"auth token on the server, OR the client used a "
                f"different token, OR the envelope is corrupted. {exc}"
            ) from exc

        try:
            im = Image.open(io.BytesIO(png_bytes))
            im.load()
        except Exception as exc:
            raise RuntimeError(
                f"PrivateDecryptLoadImage: PNG decode after decrypt "
                f"failed: {exc}") from exc

        # Build mask from alpha if present, else all-ones.
        if im.mode == "RGBA":
            alpha = np.array(im.split()[-1]).astype(np.float32) / 255.0
            mask = torch.from_numpy(alpha)[None, ...]
        else:
            h, w = im.size[1], im.size[0]
            mask = torch.ones((1, h, w), dtype=torch.float32)

        return (_pil_to_tensor(im), mask)


# ─── PrivateEncryptSaveImage ────────────────────────────────────────

class PrivateEncryptSaveImage:
    """Replaces SaveImage. Encodes the image as PNG, wraps in a V1W
    envelope, base64-encodes, and either:
      (a) writes the encrypted envelope to disk under the requested
          filename_prefix (so the file is at-rest encrypted, not
          plaintext); or
      (b) returns it via the prompt result so the client can pull it
          inline without disk involvement.

    This means even if the per-workflow privacy-cleanup fails for any
    reason, the file on disk is already AEAD-encrypted and useless to
    anyone without the auth token."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "private_enc"}),
            },
            "optional": {
                "return_inline": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE  = True
    CATEGORY     = "spellcaster/private/crypto"
    FUNCTION     = "save"

    def save(self, images, filename_prefix: str = "private_enc",
             return_inline: bool = False):
        token = _resolve_auth_token()
        key = wire_envelope.derive_key(token)

        # Resolve the output dir via ComfyUI's folder_paths helper.
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except Exception:
            output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for batch_idx in range(images.shape[0]):
            im = _tensor_to_pil(images[batch_idx:batch_idx + 1])
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            envelope = wire_envelope.wrap(
                png_bytes, key, kind=wire_envelope.KIND_IMAGE)

            if return_inline:
                # Embed in the prompt result — client pulls via /history.
                results.append({
                    "filename": f"{filename_prefix}_{batch_idx:05d}.v1w",
                    "subfolder": "",
                    "type": "output",
                    "encrypted_b64": base64.b64encode(envelope).decode(),
                })
            else:
                # Write the encrypted envelope to disk. Plaintext PNG
                # never touches output/.
                fname = f"{filename_prefix}_{batch_idx:05d}_.v1w"
                fpath = os.path.join(output_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(envelope)
                results.append({
                    "filename": fname,
                    "subfolder": "",
                    "type": "output",
                })

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "PrivateDecryptLoadImage": PrivateDecryptLoadImage,
    "PrivateEncryptSaveImage": PrivateEncryptSaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrivateDecryptLoadImage": "✧ Private Decrypt Load Image",
    "PrivateEncryptSaveImage": "✧ Private Encrypt Save Image",
}
