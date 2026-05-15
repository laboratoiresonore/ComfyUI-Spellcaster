"""pytest conftest for the ComfyUI-Spellcaster test suite.

Two duties:

  1. Put the pack root on ``sys.path`` so tests can ``import blob_bus``
     directly without going through the package's ComfyUI-coupled
     ``__init__.py`` (which would drag in `comfy.sd` etc.).
  2. Block pytest from collecting the package's own ``__init__.py`` /
     other top-level Python files as test modules — they're production
     code, not tests, and they raise ImportError on bare-import paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Pack root → sys.path.
_HERE = Path(__file__).resolve().parent
_PACK_DIR = _HERE.parent
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))

# Don't let pytest crawl the pack-root Python files (they contain
# `from .nodes.loader import ...` which fails without a parent package).
collect_ignore_glob = [
    str(_PACK_DIR / "__init__.py"),
    str(_PACK_DIR / "blob_bus.py"),
    str(_PACK_DIR / "presence.py"),
    str(_PACK_DIR / "privacy_cleanup.py"),
    str(_PACK_DIR / "private_crypto_nodes.py"),
    str(_PACK_DIR / "private_pipeline" / "*.py"),
    str(_PACK_DIR / "private_setup_route.py"),
    str(_PACK_DIR / "private_version_route.py"),
    str(_PACK_DIR / "pack_version_route.py"),
    str(_PACK_DIR / "model_repair.py"),
    str(_PACK_DIR / "install.py"),
    str(_PACK_DIR / "nodes" / "*.py"),
    str(_PACK_DIR / "spellcaster_core"),
]
