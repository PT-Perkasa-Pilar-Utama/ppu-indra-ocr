# Design

This document describes what ppu-indra-ocr does, the actors it interacts with,
and how data flows through it. It complements the API reference in the
[README](../README.md).

## What it is

ppu-indra-ocr is a library (with an optional CLI and HTTP service) that runs
Datalab's Indra OCR models: page orientation, line detection, line
recognition, reading order and layout, and optional math. A developer imports
it into their own Node, Bun, browser or React Native code and calls it
in-process. Inference runs locally through ONNX Runtime. The library does not
phone home; the only outbound traffic is the one-time model download (or none,
if models are supplied locally).

## Lifecycle

```ts
const ocr = new IndraOcrService();
await ocr.initialize(); // load the models into ONNX Runtime sessions
const page = await ocr.convert(image); // or convertPdf(ocr, pdf) from ppu-indra-ocr/pdf
await ocr.destroy(); // free the sessions and native memory
```

## Actors

| Actor                 | Role                                              | Trust                                                                           |
| :-------------------- | :------------------------------------------------ | :------------------------------------------------------------------------------ |
| Consuming application | Calls the API, supplies images, PDFs and options  | Trusted (it is the host)                                                        |
| Image or PDF input    | Bytes the application passes in                   | **Untrusted** (may be malformed)                                                |
| Model files           | Ten ONNX models and two JSON files                | Trusted from the pinned default repository; **caller-controlled** if overridden |
| JavaScript runtime    | Node, Bun, a browser engine, or React Native      | Trusted                                                                         |
| ONNX Runtime          | Runs inference (native, WASM, or mobile)          | Third-party, pinned                                                             |
| `ppu-ocv`             | Image decoding, OpenCV.js for the opencv engine   | First-party sibling, pinned                                                     |
| `@embedpdf/pdfium`    | PDF parsing and rendering (WebAssembly), optional | Third-party, pinned                                                             |

## Entry points

| Import                 | Inference backend          | Runtime      |
| :--------------------- | :------------------------- | :----------- |
| `ppu-indra-ocr`        | `onnxruntime-node`         | Node, Bun    |
| `ppu-indra-ocr/web`    | `onnxruntime-web` (WASM)   | Browser      |
| `ppu-indra-ocr/mobile` | `onnxruntime-react-native` | React Native |
| `ppu-indra-ocr/pdf`    | pdfium (WebAssembly)       | Any of above |

A CLI ships as the package `bin` (`npx ppu-indra-ocr ...`), and `apps/serve`
wraps the library as an HTTP service.

## Components

- **`src/indra/`** is the platform-free pipeline, ported from the Python
  reference: orientation and deskew, detection decode, line crops,
  recognition with script routing, line splitting, reading order and block
  clustering, markdown, math, and the PDF text-layer checks. It imports no
  runtime: sessions, image operations and PDF pages are passed in.
- **Image operations** (`ImageOps`): `opencv` (OpenCV.js, the reference's
  OpenCV release) or `canvas-native` (plain TypeScript).
- **Services** (`src/core`, `src/processor`, `src/web`, `src/mobile`) load the
  models through a platform provider and expose `convert` and the stages.
- **PDF** (`src/pdf`): pdfium reads the text layer and renders pages for OCR.
- **Model cache** stores downloads on disk (Node/Bun).

## Data flow

```
image bytes                         PDF bytes
     |                                  |
     v                                  v
 decode -> gray                    pdfium: text layer + checks
     |                                  |  trusted          | not trusted
     v                                  v                   v
 orient.onnx, deskew             render, det.onnx      render -> image path
     |                           assign characters
     v                           to detected lines
 det.onnx -> line boxes  (detect() returns here)
     |
     v
 script.onnx -> rec.onnx | rec_cjk.onnx  [+ math models]
     |
     v
 split lines, order.onnx -> reading order, block labels -> markdown
     |
     v
 Page returned to the caller
```

The library reads its input, runs inference in memory, and returns structured
pages. On Node/Bun it may read model files from disk and, on first run,
download them. It makes no other network requests (a browser also fetches the
pdfium wasm when PDF input is used) and writes only the model cache.

## Why it is split this way

The pipeline core takes its sessions, image operations and PDF pages as
arguments, so one implementation serves Node, browsers and React Native, and
the tests can check each stage against the Python reference. Each model is
its own ONNX session, as upstream ships them, so a stage can be run, replaced
or skipped on its own.
