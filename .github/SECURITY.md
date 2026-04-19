# Security Policy

## Supported Versions

Only the latest release of `ComfyUI-Spellcaster` receives security fixes. Pin a
tag only if you have a specific reason to — otherwise track `main` or the latest
tagged release.

| Version | Supported |
|---------|-----------|
| latest  | Yes       |
| older   | No        |

## Reporting a Vulnerability

If you discover a security issue in these nodes (e.g. arbitrary file read/write,
path traversal via filenames, code execution via model metadata, prompt injection
that affects downstream systems), please report it privately:

1. Open a **private vulnerability report** via
   [GitHub Security Advisories](https://github.com/laboratoiresonore/ComfyUI-Spellcaster/security/advisories/new).
2. Do not open a public issue until a fix is released.

You can expect an acknowledgement within a few days. Critical issues will be
patched and released as soon as possible.

## Scope

In scope:

- The Python nodes under `nodes/`
- The shared library under `spellcaster_core/`
- The JavaScript frontend under `web/`
- Any code this repo auto-loads from the network

Out of scope:

- Vulnerabilities in ComfyUI itself — report those at
  [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- Vulnerabilities in upstream model weights or third-party custom nodes.
