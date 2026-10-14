# Threat model

This is a security assessment of ppu-indra-ocr: where untrusted data enters,
what could go wrong on the critical paths, and how the design limits the
damage. It extends the scope section in [SECURITY.md](../SECURITY.md). Read
[DESIGN.md](DESIGN.md) first for the actors and data flow.

## Scope and assets

ppu-indra-ocr is an in-process library (plus an optional CLI and HTTP service).
It holds no secrets and no persistent state beyond an on-disk model cache. The
asset worth protecting is the **availability and integrity of the host
application** that embeds it: a bug here should not crash the host, corrupt its
memory, or let an attacker influence it through a crafted image or model path.

Publish credentials (npm, JSR) and the GHCR token for the serve image live only
in CI and are issued per-release through OIDC. No human or library code holds
them, so they are out of scope for the runtime threat model and are covered by
the release pipeline instead.

## Trust boundaries

1. **Image input -> decoder/inference.** The bytes the caller passes are
   untrusted and cross into native code (ONNX Runtime, the canvas/OpenCV
   decoder via ppu-ocv) at decode and inference time.
2. **PDF input -> pdfium (WebAssembly).** PDF bytes are parsed and rendered by
   pdfium compiled to WebAssembly. The text layer they carry is untrusted text
   that flows into the output.
3. **Model path / URL -> filesystem or network (Node).** Every model file can
   be caller-supplied. If a host forwards attacker-controlled strings, those
   reach the filesystem (path read) or a network fetch (model download).
4. **CLI / serve input -> process.** The CLI reads files named on the command
   line; the serve app accepts uploaded images and PDFs over HTTP.
5. **Dependency code -> process.** ONNX Runtime, ppu-ocv, the canvas/OpenCV
   builds and pdfium run inside the host process with its privileges.

## Critical paths and threats

| #   | Path                     | Threat                                                                                                                               | Mitigation                                                                                                                                                                                                                                                                                                                      |
| :-- | :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Image decode / inference | A malformed or hostile image triggers a crash, hang, or out-of-memory in the WASM/native runtime (denial of service)                 | Decoding and inference run in pinned upstream libraries (ONNX Runtime, ppu-ocv), updated via Dependabot. Callers should bound input size and run untrusted work off the main thread / in a sandbox. Documented in SECURITY.md.                                                                                                  |
| 2   | PDF parsing (pdfium)     | A hostile PDF exploits a pdfium bug, or a huge or many-page PDF exhausts memory and CPU                                              | pdfium runs as WebAssembly: memory-safe from the host's view, a fault surfaces as an exception. `convertPdf` turns a failed page into an error page and closes pages and documents in `finally`. Pages larger than 100 MP render at a lower dpi; `maxPages` bounds the count; serve caps it with `MAX_PDF_PAGES` and `PDF_DPI`. |
| 3   | Model path or URL (Node) | A host that passes untrusted input as a model path or URL enables path traversal, arbitrary file read, or SSRF on first-run download | The library treats these as opaque and does not sanitize them; SECURITY.md tells integrators to pass only trusted, pinned model locations. This is the host's responsibility, stated explicitly.                                                                                                                                |
| 4   | serve / CLI input        | Uploaded or named images and PDFs reach the decoders; large or crafted inputs cause resource exhaustion                              | The serve app bounds upload size, PDF pages, concurrency and queue depth (`MAX_UPLOAD_BYTES`, `MAX_PDF_PAGES`, `MAX_CONCURRENCY`, `MAX_QUEUE_DEPTH`). Treated as deployment configuration.                                                                                                                                      |
| 5   | Supply chain             | A compromised dependency or action injects code into the build or the published package                                              | Dependencies and GitHub Actions are pinned (Scorecard Pinned-Dependencies and Token-Permissions at 10), releases carry npm provenance, the published tarball runs no install scripts, CI gates on an SCA scan, and an SBOM ships with each release. See SECURITY.md.                                                            |

## What is explicitly out of scope

- Attacks that require write access to the host filesystem or the developer's
  machine.
- Vulnerabilities inside `onnxruntime-node`, `onnxruntime-web`, `ppu-ocv`, or
  pdfium themselves. Report ONNX Runtime issues to [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/security);
  we track and bump them via Dependabot.
- Misuse by the host application, such as forwarding untrusted user input as a
  model path or URL. The boundary and the host's duty are documented, but
  enforcement belongs to the host.

## Review cadence

The maintainer revisits this model when a new entry point, a new model loading
path, a new runtime dependency, or a change to the serve/CLI surface alters the
boundaries above. Material changes land through a pull request like any other.
