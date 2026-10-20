# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-20

Indra OCR 1.1: new models and the inference code of `indra-ocr` 1.1.1.
Datalab's weights license is the same file as in 1.0.

### Added

- Model presets: `model: "v1.1"` (Indra OCR 1.1, Hugging Face tag v1.1.0,
  the default) and `model: "v3.1"` (the 1.0 weights, tag v3.1). Both run on
  the 1.1 pipeline. `MODEL_PRESETS`, `DEFAULT_PRESET`, `presetFiles()` and
  `defaultModelUrl(role, preset)` describe them; the CLI takes
  `--model v1.1|v3.1`.
- Script groups: recognizers for CJK (`recCjk`), right-to-left scripts
  (`recRtl`: Arabic, Persian, Urdu, Pashto, Hebrew) and other scripts
  (`recWorld`: Indic, Thai, Lao, Khmer, Myanmar, Sinhala, Georgian,
  Armenian, Ethiopic). A line the script model is unsure of is read by the
  main recognizer too, and the more confident reading wins. Each recognizer
  loads on the first line routed to it.
- Language models: `lm` (default on in Node and Bun, off on web and React
  Native) re-reads unsure lines with a CTC beam search scored by an n-gram
  model (`lm.npz`, `lm_cjk.npz`, `lm_world.npz`, read without a new
  dependency). Each loads on the first unsure line. The CLI takes `--no-lm`.
- Right-to-left text comes out in logical order, and a right-to-left page
  reads its right column first.
- Vertical writing: a page set in vertical columns is read turned, columns
  as lines, with `page.vertical` set. `orient()` returns 4 for it, and
  `straighten(image, 4)` and `recognize(image, boxes, true)` read it by hand.
- Light text on a dark ground (banners, dark table heads) reads.
- PDF text layers: a table set sideways and a page turned by its `/Rotate`
  read upright, vertical writing reads in column order, sideways labels
  become their own lines, and right-to-left letters stored in visual order
  send the page to OCR. Characters above U+FFFF read as one character.
- `models/v1.1/`: an unmodified mirror of the 1.1 files, used by
  `PPU_INDRA_OCR_MODEL_MIRROR`.

### Changed

- **Breaking:** the default models are Indra OCR 1.1. `MODEL_REVISION` and
  `MODEL_BASE_URL` name the v1.1 preset. Pass `model: "v3.1"` for the 1.0
  weights.
- Deskew covers tilts up to 45 degrees (was 5), measured on a page preview
  before orientation. On the 8 degree test page, 0.1.2 found 4.5 degrees and
  read "1 am slightly tiltod hehe"; the 1.0 weights now read it exactly.
- Lone list markers join their item, and lines of dot leaders or fill-in
  rules are dropped (upstream `tidy_lines`).
- The order model counts a CJK character as two letters wide and knows CJK
  sentence ends.
- `IndraOcrService.downloadModels()` takes `model` and `lm` and fetches
  every file the preset can load. `ppu-indra-ocr models` lists the presets.
- `initialize()` loads about 9 MB on v1.1: the script recognizers and the
  language models load later, when a page needs them. That can be the first
  page: on the test receipt one line goes to the right-to-left recognizer
  and one through the language model, about 20 MB in the middle of
  `convert()`. Run `downloadModels()` first to avoid it.
- Pages take 50 to 180 ms longer than in 0.1.2 (Apple M1): the skew search
  covers 45 degrees, and the main recognizer reads a window three lines
  tall. The canvas-native warp now writes bytes directly to keep that down.
- Playground: a model set select and a language models switch.
- Playground: PDF pages render with their line boxes and layout blocks, like
  images (they showed text output only), and page 1 shows as soon as a PDF
  loads. The page view zooms (buttons, or Ctrl/Cmd + wheel and trackpad
  pinch) and pans by dragging when zoomed.

### Removed

- **Breaking:** `pdfOrder` and `--model-order`. Upstream 1.1 removed the
  PDF content order: the order model orders text-layer lines too.

## [0.1.2] - 2026-10-17

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

### Security

- `apps/serve`: `adm-zip` 0.6.0 -> 0.6.1 (seven advisories, among them
  GHSA-vwc7-r8mq-g2x9 and GHSA-j5f4-cc29-5x44) and `nanoid` 5.1.11 -> 5.1.16
  (GHSA-28wg-ghj8-5hjv), both transitive. The CI dependency scan now reads the
  serve lockfile too; it read only the root one.

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
