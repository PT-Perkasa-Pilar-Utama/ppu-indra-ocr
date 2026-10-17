# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The playground now has the ppu-paddle-ocr playground's design and
  workflow: drag and drop, sample files, a model download progress bar with
  an in-browser model cache, mobile views, a retry card, and a settings
  sidebar. Indra additions: pipeline and PDF settings, layout block overlays
  with a legend, Markdown, Text, Lines, Blocks and JSON tabs, per-stage
  timings, a page pager for PDFs, and a license bar.

### Fixed

- The standalone CLI binaries could not read PDFs: `pdfium.wasm` was not
  embedded. It is now, and the binary smoke test converts a PDF.
- The Linux and macOS binary archives extracted a file without the
  executable bit, so `./ppu-indra-ocr-<target>` failed with "permission
  denied". The release job now restores the bit before packing. For 0.1.1,
  run `chmod +x` on the extracted binary.

## [0.1.1] - 2026-10-17

### Fixed

- The npm upload of 0.1.0 was published from the repository root instead of
  the built package: it could not be imported and it contained the model
  weights. It is unpublished. 0.1.1 is the first working npm release; its
  code is the same as the 0.1.0 tag.

## [0.1.0] - 2026-10-17

First release: a TypeScript port of [Indra OCR](https://github.com/datalab-to/indra)
by Datalab, built on the ppu-paddle-ocr library scaffold.

### Added

- `IndraOcrService` for Node.js and Bun (`ppu-indra-ocr`), browsers
  (`ppu-indra-ocr/web`) and React Native (`ppu-indra-ocr/mobile`).
  `convert(image)` returns a `Page`: markdown, text, line boxes, layout
  blocks, rotation, skew and stage timings.
- Each stage as a method: `orient`, `straighten`, `detect`, `recognize`,
  `order`, `readMath` and `toGray`. `toMarkdown`, `joinLines` and
  `orderXYCut` as plain functions.
- `convertBatch` and `convertStream`, one page in flight by default, with
  `settle`, `signal` and `onProgress`.
- Math lines as inline LaTeX with `math: true`.
- PDF input through pdfium WebAssembly (`ppu-indra-ocr/pdf`): the text layer
  where it passes the checks, OCR elsewhere. `isPdf` from every entry.
- The `opencv` engine (matches the Python reference) and the
  `canvas-native` engine (no OpenCV, within token-F1 0.98).
- One-shot `ocr()` and `detect()`.
- Models from Datalab's Hugging Face repository pinned to commit `d7c839c`
  (tag v3.1), cached in `~/.cache/ppu-indra-ocr`, with an opt-in GitHub
  mirror (`PPU_INDRA_OCR_MODEL_MIRROR=1`). `models` reads files from paths,
  URLs or bytes.
- CLI: `convert`, `detect`, `download-models`, `clear-cache`, `models`.
- `apps/serve`: a REST server for images and PDFs with batch, stream and
  async endpoints.
- A small browser playground: images and PDFs to markdown, line boxes and
  layout blocks, with an engine switch.
- Parity tests against stage goldens from the Python `indra-ocr` 0.3.1.

### Known limits

- The web entry has automated tests only in Bun with a canvas polyfill; the
  playground was checked by hand in headless Chromium. The mobile entry is
  type-checked and has not run on a device.
- Not ported: two Python code paths the reference never runs (the table
  header heads and `balance_math`).
- The model weights are under Datalab's modified OpenRAIL-M license, not
  Apache-2.0. `LICENSE-STATUS.md` explains the terms, records our
  unanswered 2026-10-14 request to Datalab, and lists what may change.
