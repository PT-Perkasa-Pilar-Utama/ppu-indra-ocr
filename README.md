# ppu-indra-ocr

[![NPM](https://img.shields.io/npm/dw/ppu-indra-ocr)](https://www.npmjs.com/package/ppu-indra-ocr) [![npm version](https://img.shields.io/npm/v/ppu-indra-ocr)](https://www.npmjs.com/package/ppu-indra-ocr) [![JSR](https://jsr.io/badges/@snowfluke/ppu-indra-ocr)](https://jsr.io/@snowfluke/ppu-indra-ocr) [![Provenance](https://img.shields.io/badge/npm-signed%20provenance-blue?logo=npm)](https://www.npmjs.com/package/ppu-indra-ocr#provenance) [![License: Apache-2.0](https://img.shields.io/npm/l/ppu-indra-ocr)](./LICENSE) [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/PT-Perkasa-Pilar-Utama/ppu-indra-ocr/badge)](https://scorecard.dev/viewer/?uri=github.com/PT-Perkasa-Pilar-Utama/ppu-indra-ocr) [![Socket Badge](https://socket.dev/api/badge/npm/package/ppu-indra-ocr)](https://socket.dev/npm/package/ppu-indra-ocr)

Indra OCR in TypeScript. Images and PDFs in; markdown, plain text, line boxes and layout blocks out. The whole [Indra OCR](https://github.com/datalab-to/indra) pipeline by Datalab runs on ONNX Runtime: page orientation, deskew, line detection, recognition, reading order, layout and markdown. Runs anywhere JavaScript runs: Node.js, Bun, Deno, web browsers, web workers, browser extensions, and React Native (iOS/Android). Docker & CLI supported. [Try it in your browser](https://ppu-indra-ocr.snowfluke.workers.dev/).

Need it as an HTTP service? Dockerized? Spin up the REST API here: [apps/serve](/apps/serve/README.md). Need a CLI instead? See [Command Line](#command-line). Are you an AI agent? Learn the API quickly with the skill in the `skill-ppu-indra-ocr` folder.

> [!WARNING]
> **Read Datalab's license before you use this package.**
>
> This package runs the Indra OCR model weights by Datalab. The weights are
> not under this package's Apache-2.0 license. They are under Datalab's
> modified OpenRAIL-M license, and you accept it when the library downloads
> them for you. You may not use the weights at all if:
>
> - your organization had more than USD 20M in revenue last year, or has
>   raised more than USD 20M, unless your use is personal or research; or
> - your organization offers any product or service that competes with
>   Datalab. This clause has no exception for personal use or research.
>
> You are responsible for your own compliance. Read
> [LICENSE-STATUS.md](LICENSE-STATUS.md) for the terms, our open request to
> Datalab, and what may change. This is a community project. Datalab does not
> endorse it.

![ppu-indra-ocr demo](https://raw.githubusercontent.com/PT-Perkasa-Pilar-Utama/ppu-indra-ocr/refs/heads/main/assets/ppu-indra-ocr-demo.jpg)

```ts
import { IndraOcrService } from "ppu-indra-ocr";

const service = new IndraOcrService();
await service.initialize();

const page = await service.convert(await Bun.file("scan.png").arrayBuffer());
console.log(page.markdown);

await service.destroy();
```

## Table of Contents

- [Why ppu-indra-ocr?](#why-ppu-indra-ocr)
- [Runtime Support](#runtime-support)
- [Installation](#installation)
  - [CLI (global install)](#cli-global-install)
  - [Standalone Binaries](#standalone-binaries)
- [Core Usage](#core-usage)
  - [Basic Conversion](#basic-conversion)
  - [The Page Result](#the-page-result)
  - [Running the Stages Yourself](#running-the-stages-yourself)
  - [Custom Models](#custom-models)
  - [Detection Only](#detection-only)
- [Command Line](#command-line)
- [Batch Conversion](#batch-conversion)
- [PDF Input](#pdf-input)
- [Math](#math)
- [Orientation and Deskew](#orientation-and-deskew)
- [Processing Engine](#processing-engine)
- [Web / Browser Support](#web--browser-support)
  - [Using a Bundler](#using-a-bundler-vite-webpack-etc)
  - [Web Workers](#web-workers)
  - [CDN (No Bundler)](#cdn-no-bundler)
  - [WebGPU Acceleration](#webgpu-acceleration)
  - [Multithreaded WASM (Cross-Origin Isolation)](#multithreaded-wasm-cross-origin-isolation)
- [React Native (Mobile)](#react-native-mobile)
- [Models and Language Support](#models-and-language-support)
  - [Indra OCR Models](#indra-ocr-models)
  - [Cache Location](#cache-location-node--bun)
  - [Languages](#languages)
  - [Model Output Limitations](#model-output-limitations)
- [Configuration Reference](#configuration-reference)
  - [IndraOptions](#indraoptions)
  - [ModelSources](#modelsources)
  - [BatchOptions](#batchoptions)
  - [ConvertPdfOptions](#convertpdfoptions)
  - [SessionOptions](#sessionoptions)
  - [Model host mirror](#model-host-mirror)
- [Benchmark](#benchmark)
- [How the Port Is Checked](#how-the-port-is-checked)
- [Ecosystem](#ecosystem)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)
- [Scripts](#scripts)

## Why ppu-indra-ocr?

- **The whole page, not just the text.** Orientation, deskew, line detection, Latin and Chinese recognizers, reading order, layout labels, and markdown with headings, lists and tables, in one `convert()` call.
- **PDFs too.** Uses a PDF's text layer where it can be trusted and OCR where it cannot, page by page.
- **Small models.** About 18 MB of ONNX models (38 MB with math), fetched and cached on first run, pinned to one upstream revision.
- **Matches the reference.** On the `opencv` engine, line text, layout blocks and markdown match the Python `indra-ocr` exactly on every test page. See [How the Port Is Checked](#how-the-port-is-checked).
- **Runs everywhere**, Node.js, Bun, Deno, web browsers, web workers, browser extensions, and React Native (iOS/Android), on the same runtime adapters as [ppu-paddle-ocr](https://github.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr).
- **Composable.** Every stage is a method, so you can run one stage, swap a stage, or chain your own pipeline.
- **TypeScript**, full type definitions.

## Runtime Support

The same package, the same API, every JavaScript runtime:

| Runtime                          | How to install                                                                                             | Try it                                                     |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Node.js**                      | `npm install ppu-indra-ocr onnxruntime-node`                                                               | [npm package](https://www.npmjs.com/package/ppu-indra-ocr) |
| **Bun**                          | `bun add ppu-indra-ocr onnxruntime-node`                                                                   | [npm package](https://www.npmjs.com/package/ppu-indra-ocr) |
| **Deno**                         | `deno add jsr:@snowfluke/ppu-indra-ocr`                                                                    | [JSR package](https://jsr.io/@snowfluke/ppu-indra-ocr)     |
| **Web browser**                  | `npm install ppu-indra-ocr onnxruntime-web` (import `/web` subpath)                                        | [Live demo](https://ppu-indra-ocr.snowfluke.workers.dev/)  |
| **Web worker**                   | Same as web; the `/web` subpath runs in workers.                                                           | [Web Workers](#web-workers)                                |
| **Browser extension**            | Same as web; bundle `ppu-indra-ocr/web` with your extension's bundler.                                     | [Web / Browser Support](#web--browser-support)             |
| **Mobile (React Native)**        | `npm install ppu-indra-ocr onnxruntime-react-native @shopify/react-native-skia` (import `/mobile` subpath) | [React Native](#react-native-mobile)                       |
| **PDF input** (any of the above) | `npm install @embedpdf/pdfium` (import `/pdf` subpath)                                                     | [PDF Input](#pdf-input)                                    |

## Installation

```bash
npm install ppu-indra-ocr onnxruntime-node onnxruntime-web
```

Omit `onnxruntime-node` or `onnxruntime-web` depending on your target environment (Node/Bun vs browser). Add `@embedpdf/pdfium` for PDF input.

The first `initialize()` downloads the models (about 18 MB, 38 MB with math) from Datalab's Hugging Face repository, pinned to one revision, and caches them in `~/.cache/ppu-indra-ocr` on Node and Bun.

**When the models download, you accept Datalab's license for the weights.**
Check [LICENSE-STATUS.md](LICENSE-STATUS.md) first.

### CLI (global install)

```bash
npm install -g ppu-indra-ocr onnxruntime-node      # or: bun add -g ppu-indra-ocr onnxruntime-node
ppu-indra-ocr convert scan.png
```

This puts a `ppu-indra-ocr` command on your `PATH`. See [Command Line](#command-line) for all commands.

<details>
<summary>Notes on the global install</summary>

- **`onnxruntime-node` is required.** It is an optional peer dependency (native binaries), so install it alongside the CLI. Add `@embedpdf/pdfium` globally too if you want PDF input.
- **bun**: ensure `~/.bun/bin` is on your `PATH` (npm's global bin usually already is).
- **Updates are manual.** Re-run the install with `@latest` to upgrade.
- **Node or Bun must be present.** This is the Node/Bun build, not a standalone binary. For that, see [Standalone Binaries](#standalone-binaries) below.

</details>

### Standalone Binaries

Every release attaches self-contained executables, no Node, Bun, or `npm install` required. Download, extract, run:

```bash
curl -L https://github.com/PT-Perkasa-Pilar-Utama/ppu-indra-ocr/releases/latest/download/ppu-indra-ocr-linux-x64.tar.gz | tar xz
./ppu-indra-ocr-linux-x64 convert scan.png
```

| Asset                               | Platform                                                       |
| :---------------------------------- | :------------------------------------------------------------- |
| `ppu-indra-ocr-linux-x64.tar.gz`    | Linux x86-64 (glibc; Debian, Ubuntu, Fedora, ...)              |
| `ppu-indra-ocr-linux-arm64.tar.gz`  | Linux ARM64 (glibc; Raspberry Pi OS 64-bit, AWS Graviton, ...) |
| `ppu-indra-ocr-darwin-arm64.tar.gz` | macOS on Apple Silicon                                         |
| `ppu-indra-ocr-windows-x64.zip`     | Windows x86-64 (contains the `.exe`)                           |

Intel Macs and Alpine (musl) are not covered, onnxruntime ships no darwin-x64 or musl builds; use `npx`/`bunx` there.

<details>
<summary>Size, slim variant, and OS warnings</summary>

- **Size and first run.** 53-69 MB download depending on the platform, about 165 MB extracted on macOS (the executable embeds the Bun runtime, ONNX Runtime, pdfium and the image engines). Models (about 18 MB) still download to `~/.cache/ppu-indra-ocr` on first run, exactly like the npm CLI.
- **Slim variant.** Each target also ships a `-slim` archive (e.g. `ppu-indra-ocr-linux-x64-slim.tar.gz`) that excludes the OpenCV engine: about 30 MB smaller extracted, 4-5 MB smaller to download. It always uses the `canvas-native` engine and rejects `--engine opencv`. Pick slim for disk and memory footprint; pick full if you want results that match the Python reference exactly.
- **macOS Gatekeeper.** The binaries are ad-hoc signed but not Apple-notarized. If macOS blocks a downloaded binary, clear the quarantine flag: `xattr -d com.apple.quarantine ./ppu-indra-ocr-darwin-arm64` (or right-click, Open, once).
- **Windows SmartScreen.** Unrecognized-app warning on first run: "More info", then "Run anyway".

</details>

<details>
<summary>Verify a download (Sigstore + SLSA)</summary>

Both checks run against the archive, before extraction; both artifacts are attached per release.

```bash
# Sigstore signature (bundle is attached next to each archive)
cosign verify-blob ppu-indra-ocr-linux-x64.tar.gz \
  --bundle ppu-indra-ocr-linux-x64.tar.gz.sigstore.json \
  --certificate-identity-regexp "github.com/PT-Perkasa-Pilar-Utama/ppu-indra-ocr" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com

# GitHub build provenance (SLSA)
gh attestation verify ppu-indra-ocr-linux-x64.tar.gz --repo PT-Perkasa-Pilar-Utama/ppu-indra-ocr
```

</details>

## Core Usage

### Basic Conversion

```ts
import { IndraOcrService } from "ppu-indra-ocr";

const service = new IndraOcrService({ verbose: true });
await service.initialize();

const page = await service.convert(await Bun.file("./assets/receipt.jpg").arrayBuffer());
console.log(page.markdown);

await service.destroy();
```

`convert` takes encoded image bytes (`ArrayBuffer` or `Uint8Array`: PNG, JPEG, WebP and the other formats your canvas decodes), a canvas, or gray pixels `{ data: Uint8Array, width, height }`.

For a single image, `ocr()` and `detect()` manage a short-lived service for you. Create an `IndraOcrService` when processing more than one image so the models remain initialized between calls.

```ts
import { detect, ocr } from "ppu-indra-ocr";

const page = await ocr(imageBytes);
const boxes = await detect(imageBytes); // [x0, y0, x1, y1, score][]
```

### The Page Result

```ts
type Page = {
  markdown: string | null; // headings, paragraphs, lists, tables as HTML
  text: string; // every line, one per row, in reading order
  lines: { box: [x0, y0, x1, y1]; text: string; conf: number }[];
  blocks: { label: BlockLabel; bbox: [x0, y0, x1, y1]; text: string }[];
  source: "ocr" | "text" | "hybrid" | "error"; // text / hybrid: from a PDF text layer
  note: string; // why the page took its path
  rotation: number; // degrees counter-clockwise the input was turned
  skew: number; // degrees of small tilt undone
  timings: Record<string, number>; // milliseconds per stage, with total_ms
};
```

Block labels: `Text`, `Section-Header`, `Page-Header`, `Page-Footer`, `List-Group`, `Table`, `Caption`, `Footnote`, `Table-Of-Contents`, `Form`. Page headers and footers are left out of the markdown unless you pass `furniture: true`; they stay in `lines` and `blocks`. Boxes are in the upright page's pixels.

### Running the Stages Yourself

```ts
const { turns } = await service.orient(image); // quarter turns, from orient.onnx
const { image: upright, skew } = await service.straighten(image, turns);
const boxes = (await service.detect(upright)).map(([x0, y0, x1, y1]) => [x0, y0, x1, y1] as const);
const lines = await service.recognize(upright, boxes); // text, confidence, character positions
const texts = lines.map((l) => l.text);
const { order, cls, same } = await service.order(boxes, texts, upright.width, upright.height);
```

`recognize` takes boxes from any detector, one line per box. Boxes should hug the text: the recognizer adds 0.2 x the box height above and below and 0.3 x on each side. Trim boxes that carry their own padding first; PaddleOCR's boxes, for example, are padded to 1.8 x the text height.

`toMarkdown`, `joinLines` and `orderXYCut` are exported as plain functions. `service.pipeline` gives the lower-level stages (line splitting, assembly) when you need them. `readMath(image, boxes)` reads boxes as text with inline LaTeX on a service created with `math: true`.

### Custom Models

Every model file has a role. Pass `models` to read some or all of them from file paths (Node/Bun), URLs, or `ArrayBuffer`s; files you leave out come from the default repository:

```ts
import { IndraOcrService, MODEL_FILES } from "ppu-indra-ocr";

// One file from elsewhere
const service = new IndraOcrService({
  models: { det: "./models/det.onnx" },
});

// Every file from a local folder (air-gapped hosts)
const local = new IndraOcrService({
  models: Object.fromEntries(
    Object.entries(MODEL_FILES).map(([role, file]) => [role, `/opt/indra-models/${file}`])
  ),
});
```

Roles: `det`, `rec`, `recCjk`, `script`, `orient`, `order`, `meta`, and with math `mathCls`, `mathEnc`, `mathDecInit`, `mathDecStep`, `mathMeta`. `MODEL_FILES` maps each role to its file name, `MODEL_REVISION` is the pinned upstream commit, and `defaultModelUrl(role)` gives the default URL.

### Detection Only

`detect()` runs only the detection model, no recognition, and returns the line boxes it finds with their scores. Useful when you only need layout (where the text is), or want to feed the boxes into your own pipeline.

```ts
const boxes = await service.detect(imageBytes);
// boxes: [x0, y0, x1, y1, score][] in image pixels
```

`detect()` runs on the image as given. Run `orient` and `straighten` first when the input may be turned or tilted. Also available as the CLI `detect` command and the serve app's `POST /v1/detect`, see [Command Line](#command-line) and [`apps/serve`](/apps/serve/README.md).

## Command Line

The package ships a `bin`, so you can OCR without writing any code. In a project that has `ppu-indra-ocr` and `onnxruntime-node` installed, `bunx`/`npx` resolve the local install directly; for zero-install runs use `npx -p onnxruntime-node -p ppu-indra-ocr ppu-indra-ocr <args>` or a [global install](#cli-global-install):

```bash
# one image or PDF -> markdown on stdout
bunx ppu-indra-ocr convert paper.pdf

# many images (glob), one output file per input
bunx ppu-indra-ocr convert "scans/*.png" --out results/

# structured JSON, one object per page
bunx ppu-indra-ocr convert scan.png --format json --pretty

# force OCR on a PDF, read math as LaTeX, print per-page timings
bunx ppu-indra-ocr convert paper.pdf --mode ocr --math --timings

# detection only: line boxes as JSON
bunx ppu-indra-ocr detect receipt.jpg

# zero-install (no local ppu-indra-ocr): npx can pull both packages
npx -p onnxruntime-node -p ppu-indra-ocr ppu-indra-ocr convert scan.png

# pre-warm / clear the model cache, list the model files and revision
bunx ppu-indra-ocr download-models
bunx ppu-indra-ocr clear-cache
bunx ppu-indra-ocr models
```

| Flags                                                           | Applies to   | Purpose                                                       |
| :-------------------------------------------------------------- | :----------- | :------------------------------------------------------------ |
| `--format md\|text\|json`, `--out <dir>`                        | `convert`    | Output format; one file per input (`.md`, `.txt`, `.jsonl`)   |
| `--math`, `--keep-furniture`, `--engine`, `--timings`           | `convert`    | Math as LaTeX, keep headers and footers, engine, timings      |
| `--mode auto\|ocr\|text`, `--dpi`, `--max-pages`, `--password`  | PDFs         | Text layer or OCR, render resolution, page cap, password      |
| `--model-order`, `--no-image-ocr`                               | PDFs         | Order text-layer lines with the model; skip OCR inside images |
| `--model-dir <dir>`, `--execution-providers <list>`             | all commands | Read models from a folder; ONNX providers (e.g. `cuda,cpu`)   |
| `-o`/`--output <file>`, `--pretty`, `-q`/`--quiet`, `--verbose` | all commands | Output destination and format, logging                        |

Converted text goes to **stdout**; progress and logs go to **stderr**, so output pipes cleanly. Exit codes: `0` success, `1` runtime error, `2` usage error.

Run `bunx ppu-indra-ocr help` for the full reference.

## Batch Conversion

`convertBatch()` runs `convert()` over many images with **bounded concurrency**: at most `concurrency` images are decoded and in flight at once, so memory stays in check. Results come back **index-aligned** to the inputs, whatever order they finish in.

```ts
const pages = await service.convertBatch([buf1, buf2, buf3]);
pages.forEach((p, i) => console.log(i, p.markdown));
```

| You want                            | Use                                   |
| :---------------------------------- | :------------------------------------ |
| More images in flight               | `{ concurrency: 2 }`                  |
| One bad image not to kill the batch | `{ settle: true }`                    |
| Progress or cancellation            | `{ onProgress, signal }`              |
| Results as they finish              | `convertStream()`                     |
| An input list too big for memory    | pass an `Iterable` or `AsyncIterable` |

Full surface: [`BatchOptions`](#batchoptions).

**Concurrency.** Defaults to `1`. Each page allocates large tensors (a whole page goes through detection at a 1024 px long side, then layout and reading order), so more pages at once mostly add memory. Raise it only after measuring on your hardware.

**Failures.** With `settle: true` the call never rejects; each slot becomes `{ index, status, value | reason }`:

```ts
const results = await service.convertBatch(images, { settle: true });
for (const r of results) {
  if (r.status === "fulfilled") console.log(r.value.markdown);
  else console.error("failed:", r.reason);
}
```

**Progress and cancellation** use the usual primitives:

```ts
const ac = new AbortController();
await service.convertBatch(images, {
  signal: ac.signal,
  onProgress: (done, total) => console.log(`${done}/${total}`),
});
```

**Streaming.** `convertStream()` yields each result as it finishes, so the whole batch is never buffered. Each item carries its input `index` for reordering:

```ts
for await (const item of service.convertStream(images)) {
  if (item.status === "fulfilled") console.log(item.index, item.value.markdown);
}
```

Both methods accept any `Iterable` or `AsyncIterable` of inputs, so a directory walk or a queue never has to be materialized in memory at once.

## PDF Input

```ts
import { IndraOcrService } from "ppu-indra-ocr";
import { convertPdf, isPdf } from "ppu-indra-ocr/pdf";

const service = new IndraOcrService();
await service.initialize();

const pages = isPdf(bytes)
  ? await convertPdf(service, bytes, { mode: "auto" })
  : [await service.convert(bytes)];
```

| Option     | Default  | Meaning                                                                                                                               |
| ---------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`     | `"auto"` | `auto`: the text layer where it passes the checks, OCR elsewhere. `ocr`: always OCR. `text`: the text layer even when the checks fail |
| `dpi`      | `200`    | Render resolution for OCR                                                                                                             |
| `pdfOrder` | `true`   | Keep the PDF content order when it is coherent                                                                                        |
| `imageOcr` | `true`   | On text-layer pages, OCR lines with no text-layer characters (text in images)                                                         |
| `maxPages` | all      | Read at most this many pages                                                                                                          |
| `password` | -        | Password of an encrypted PDF                                                                                                          |

The text layer is checked first: invisible text (an OCR layer), broken font maps, and runs with no ink under them send the page to OCR. A page that fails comes back with `source: "error"` and the reason in `note`; a file that cannot be opened throws. pdfium runs as WebAssembly (`@embedpdf/pdfium`); a browser fetches its wasm from jsDelivr unless you pass `pdfium: await loadPdfium(wasmBytesOrUrl)`. `isPdf(bytes)` is exported from every entry to route files without loading pdfium. On Node and Bun, set `PPU_INDRA_OCR_PDFIUM_WASM` to a local `pdfium.wasm` path to load it from there instead of `node_modules`.

## Math

```ts
const service = new IndraOcrService({ math: true });
```

A classifier finds lines that hold math, and an attention decoder reads them as text with inline LaTeX (`we have $x _ { i } \leq 1$`). A reading replaces a line only when it holds math and its mean token probability is 0.9 or more. Math loads four more models and is slower on pages that have it.

## Orientation and Deskew

There is nothing to compose: `convert()` turns the page upright with `orient.onnx` (90, 180 and 270 degree turns) and undoes small tilt before detection. `page.rotation` and `page.skew` report what was undone, and boxes come back in the upright page's pixels. Call `service.orient()` and `service.straighten()` directly when you want those stages alone. For general image preprocessing (grayscale, thresholding, contours), see [ppu-ocv](https://github.com/PT-Perkasa-Pilar-Utama/ppu-ocv).

## Processing Engine

Two image processing backends are available for resizing and warping:

| Engine            |   Default    | OpenCV Required | Notes                                                                  |
| :---------------- | :----------: | :-------------: | :--------------------------------------------------------------------- |
| `"opencv"`        | Node and Bun |       Yes       | OpenCV.js through `ppu-ocv`. Results match the Python reference.       |
| `"canvas-native"` | Web, mobile  |       No        | Plain TypeScript. Smaller; within a token-F1 of 0.98 of the reference. |

```ts
// OpenCV (default on Node and Bun)
const service = new IndraOcrService();

// Canvas-native (no OpenCV dependency)
const service = new IndraOcrService({ engine: "canvas-native" });
```

The opencv engine runs the same OpenCV release (5.0) as the reference. On the web, `engine: "opencv"` needs OpenCV.js on the page first; the library does not fetch it (about 10 MB):

```html
<script src="https://cdn.jsdelivr.net/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js"></script>
```

Without OpenCV.js on `globalThis.cv`, `initialize()` throws and says so.

## Web / Browser Support

Import from `ppu-indra-ocr/web` for browser-native capabilities (`HTMLCanvasElement`, `OffscreenCanvas`, `fetch` buffering).

### Using a Bundler (Vite, Webpack, etc.)

```ts
import { IndraOcrService } from "ppu-indra-ocr/web";

const service = new IndraOcrService(); // canvas-native, wasm
await service.initialize();

const file = document.getElementById("upload").files[0];
const page = await service.convert(await file.arrayBuffer());
console.log(page.markdown);
```

Models are fetched from the default URLs or from `models`.

### Web Workers

Keep the service inside a worker and send image bytes across the boundary, so inference never blocks the main thread:

```ts
// worker.ts
import { IndraOcrService } from "ppu-indra-ocr/web";

const service = new IndraOcrService();
const ready = service.initialize();

self.onmessage = async (event: MessageEvent<ArrayBuffer>) => {
  await ready;
  const page = await service.convert(event.data);
  self.postMessage(page);
};
```

```ts
// main.ts
const worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});
worker.onmessage = (event) => console.log(event.data.markdown);

const bytes = await file.arrayBuffer();
worker.postMessage(bytes, [bytes]); // transferred, not copied
```

Notes:

- A worker is the right home for OCR. On the main thread, inference blocks the page until the whole image is done.
- Cross-origin isolation still governs WASM threading, see [Multithreaded WASM](#multithreaded-wasm-cross-origin-isolation). It is a property of the page, not of the worker.
- `isWebWorker()` is exported if you need to branch on the scope yourself.

```ts
import { isWebWorker } from "ppu-indra-ocr/web";
```

### CDN (No Bundler)

See the [live demo](https://ppu-indra-ocr.snowfluke.workers.dev/) for a complete ESM/CDN setup with an import map.

### WebGPU Acceleration

ONNX Runtime runs on WASM unless you ask for more. To try WebGPU, opt in with the providers the browser supports:

```ts
import {
  IndraOcrService,
  getDefaultWebExecutionProviders,
  isWebGpuAvailable,
} from "ppu-indra-ocr/web";

if (await isWebGpuAvailable()) console.log("WebGPU supported");

const service = new IndraOcrService({
  session: { executionProviders: await getDefaultWebExecutionProviders() }, // ["webgpu", "wasm"] or ["wasm"]
});
```

If session creation with the requested providers fails, the service rebuilds the session on `wasm`; pass `session.onSessionFallback` to be told.

**The WASM binaries are always required**, even when WebGPU is the primary provider. When `ort.env.wasm.wasmPaths` is unset, pages and workers load the jsDelivr copy of the exact `onnxruntime-web` version you loaded, so the loader and the binaries never disagree. To self-host them, set it before `initialize()`:

```ts
ort.env.wasm.wasmPaths = "/ort/"; // your copy of the onnxruntime-web dist files
```

### Multithreaded WASM (Cross-Origin Isolation)

ONNX Runtime runs WASM multithreaded only if the page is [cross-origin isolated](https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated); otherwise `numThreads` is pinned to 1. Isolation needs two response headers:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

**If you can set those headers server-side, do that.** It is the correct fix and needs nothing from this package.

For static hosts that cannot (e.g. GitHub Pages), the package ships an opt-in [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) that injects the headers client-side. Copy it to your served root and load it **before** anything else:

```html
<script src="/coi-serviceworker.js"></script>
```

Resolve the shipped copy from the package, e.g. in a build step:

```ts
// path on disk: node_modules/ppu-indra-ocr/coi-serviceworker.js
const swPath = import.meta.resolve("ppu-indra-ocr/coi-serviceworker.js");
```

> The service worker reloads the page once on first visit to apply the headers, and rewrites all fetch responses. Don't use it if you already control your headers or run another service worker that conflicts.

## React Native (Mobile)

Run the same OCR pipeline on iOS and Android via the `ppu-indra-ocr/mobile` entry. It uses `onnxruntime-react-native` (native JSI inference) and `ppu-ocv/canvas-mobile` (Skia-backed canvas) instead of their web counterparts, with the `canvas-native` engine.

```bash
npm install ppu-indra-ocr onnxruntime-react-native @shopify/react-native-skia
```

```ts
import { IndraOcrService } from "ppu-indra-ocr/mobile";

const service = new IndraOcrService();
await service.initialize();

// `imageBuffer` is an ArrayBuffer e.g. from a captured frame or a bundled asset.
const page = await service.convert(imageBuffer);
console.log(page.markdown);

await service.destroy();
```

**Requirements.** RN >= 0.74 / Expo SDK >= 51 (Hermes). Both peer packages ship native code, so you need a dev client or `expo prebuild`. **Expo Go is not supported.**

**Hardware acceleration is opt-in**, mobile inference runs on CPU by default. There is no WebGPU on React Native:

```ts
// Android: ["nnapi"]. iOS: ["coreml"].
new IndraOcrService({ session: { executionProviders: ["nnapi"] } });
```

**Camera capture is out of scope.** Pass a decoded frame from `react-native-vision-camera` or `expo-camera` as an `ArrayBuffer`.

<details>
<summary>Why the mobile entry is on npm but not JSR</summary>

`onnxruntime-react-native` and `@shopify/react-native-skia` are native React Native modules that only resolve through npm/Metro, and JSR's Deno-based publish cannot resolve them (React Native itself cannot consume JSR packages). So the mobile entry is excluded from the JSR package. Install via npm/yarn/bun as shown above.

</details>

## Models and Language Support

### Indra OCR Models

Indra OCR is a small CPU pipeline with a 1M-parameter line recognizer. The default models come from [datalab-to/indra-ocr](https://huggingface.co/datalab-to/indra-ocr), pinned to commit `d7c839c` (tag v3.1), downloaded and cached on first run:

| Role     | File           | Size   | What it does                                    |
| :------- | :------------- | :----- | :---------------------------------------------- |
| `det`    | `det.onnx`     | 0.4 MB | Text line detection                             |
| `rec`    | `rec.onnx`     | 5.0 MB | Line recognizer (Latin, Cyrillic, Greek, Indic) |
| `recCjk` | `rec_cjk.onnx` | 9.3 MB | Chinese line recognizer                         |
| `script` | `script.onnx`  | 0.2 MB | Script classifier (routes Chinese lines)        |
| `orient` | `orient.onnx`  | 0.4 MB | Page orientation                                |
| `order`  | `order.onnx`   | 3.0 MB | Reading order, layout labels and blocks         |
| `meta`   | `meta.json`    | 0.1 MB | Vocabularies and input settings                 |

With `math: true`, five more files load: `mathcls.onnx`, `math_enc.onnx`, `math_dec_init.onnx`, `math_dec_step.onnx` and `math_meta.json` (about 20 MB).

The weights are Datalab's, under their own license. Unmodified mirrors with the license and attribution: `models/` in this repository and [snowfluke/ppu-indra-ocr-models](https://huggingface.co/snowfluke/ppu-indra-ocr-models) on Hugging Face.

### Cache Location (Node / Bun)

Models are cached under `~/.cache/ppu-indra-ocr`:

| OS      | Path                                       |
| :------ | :----------------------------------------- |
| macOS   | `~/.cache/ppu-indra-ocr`                   |
| Linux   | `~/.cache/ppu-indra-ocr`                   |
| Windows | `C:\Users\<username>\.cache\ppu-indra-ocr` |

Each file lands in a subdirectory named after a digest of its source URL, so two models that share a file name never collide.

```ts
// Warm the cache (e.g. in CI or Docker builds)
await IndraOcrService.downloadModels({ math: false });
```

Clear it with `ppu-indra-ocr clear-cache`.

> In the browser, model files are fetched via `fetch()` on every page load and rely on the browser's HTTP cache. For persistent offline caching, use a Service Worker or store the `ArrayBuffer` in IndexedDB and pass it through `models`.

### Languages

There is no language option: a script model routes Chinese lines to the CJK recognizer, and everything else goes to the main recognizer.

- **OCR (scans and images):** Latin (about 50 languages), Cyrillic, Greek, Devanagari, Bengali, and Chinese (simplified and traditional).
- **PDF text layer:** any script.

### Model Output Limitations

From the upstream model card: Indra OCR is built for speed on clean print. Complex tables (merged or spanning cells, no ruling lines), handwriting, scripts outside the OCR list (Arabic, Hebrew, Thai, Japanese kana, Korean on scans), degraded scans, vertical text and very wide pages read poorly.

In this port, the right-to-left check for PDF text-layer lines uses script ranges instead of Unicode bidi classes.

## Configuration Reference

### `IndraOptions`

```ts
import type { IndraOptions } from "ppu-indra-ocr";

export type IndraOptions = {
  models?: ModelSources;
  math?: boolean;
  furniture?: boolean;
  adaptive?: boolean;
  engine?: "opencv" | "canvas-native";
  session?: SessionOptions;
  verbose?: boolean;
};
```

| Property    |             Type              |                    Default                     | Description                                    |
| :---------- | :---------------------------: | :--------------------------------------------: | :--------------------------------------------- |
| `models`    |        `ModelSources`         |          Datalab's repository, pinned          | Per-file sources, see below                    |
| `math`      |           `boolean`           |                    `false`                     | Read math lines as LaTeX (four more models)    |
| `furniture` |           `boolean`           |                    `false`                     | Keep page headers and footers in the markdown  |
| `adaptive`  |           `boolean`           |                     `true`                     | Second detection pass on pages with small text |
| `engine`    | `"opencv" \| "canvas-native"` | `"opencv"` on Node/Bun, else `"canvas-native"` | Image engine (see above)                       |
| `session`   |       `SessionOptions`        |           CPU, CPU memory arena off            | ONNX Runtime session options                   |
| `verbose`   |           `boolean`           |                    `false`                     | Log loading and stage steps to the console     |

### `ModelSources`

A partial map from file role to a source: a URL, a file path (Node/Bun only), or an `ArrayBuffer`. Roles left out use `defaultModelUrl(role)`.

| Property                                                       |          Type           |    Default     | Description                  |
| :------------------------------------------------------------- | :---------------------: | :------------: | :--------------------------- |
| `det`, `rec`, `recCjk`, `script`, `orient`, `order`, `meta`    | `string \| ArrayBuffer` | pinned default | The base files               |
| `mathCls`, `mathEnc`, `mathDecInit`, `mathDecStep`, `mathMeta` | `string \| ArrayBuffer` | pinned default | The math files (with `math`) |

### `BatchOptions`

Options of `convertBatch()` / `convertStream()`.

| Property      |           Type           | Default | Description                                                                              |
| :------------ | :----------------------: | :-----: | :--------------------------------------------------------------------------------------- |
| `concurrency` |         `number`         |   `1`   | Max images in flight.                                                                    |
| `settle`      |        `boolean`         | `false` | When `true`, a failed image yields `{ status: "rejected", reason }` instead of throwing. |
| `signal`      |      `AbortSignal`       |    -    | Cancels the batch; pending images are not scheduled and the call rejects.                |
| `onProgress`  | `(done, total?) => void` |    -    | Called after each image settles, with the running count and total (if known).            |

### `ConvertPdfOptions`

Options of `convertPdf()`: `mode`, `dpi`, `pdfOrder`, `imageOcr`, `maxPages` and `password` (see [PDF Input](#pdf-input)), plus `pdfium`, a module from `loadPdfium()` that is otherwise loaded once and reused. `PDFIUM_VERSION` names the `@embedpdf/pdfium` release the package is tested with.

### `SessionOptions`

Any valid ONNX Runtime `InferenceSession.SessionOptions` property is accepted. ppu-indra-ocr changes one default:

| Property             |                  Type                   | Default | Description                                                                          |
| :------------------- | :-------------------------------------: | :-----: | :----------------------------------------------------------------------------------- |
| `executionProviders` | `string[] \| ExecutionProviderConfig[]` |  ORT's  | Execution providers for inference (CPU on Node/Bun, WASM on the web).                |
| `enableCpuMemArena`  |                `boolean`                | `false` | Off: the arena keeps each page's peak memory; turning it on trades memory for speed. |
| `onSessionFallback`  |       `(error: unknown) => void`        |    -    | Called when the requested providers fail and the session is rebuilt on `cpu`/`wasm`. |

`onSessionFallback` is the only non-ORT property. It is stripped before the options reach ONNX Runtime. Use it when you run your own accelerator ladder and need to know that a GPU provider was dropped instead of reading a log line:

```ts
let degraded: unknown = null;
const service = new IndraOcrService({
  session: {
    executionProviders: [{ name: "cuda", deviceId: 0 }, "cpu"],
    onSessionFallback: (error) => (degraded ||= error),
  },
});
await service.initialize();
if (degraded) {
  // Acceleration was dropped. Fail over to your own runtime profile.
}
```

It can fire once per model session.

### Model host mirror

Built-in models download from Datalab's Hugging Face repository. That host rate-limits by IP, which shows up as `HTTP 429 Too Many Requests` from shared addresses such as CI runners.

Set `PPU_INDRA_OCR_MODEL_MIRROR=1` and a download that exhausts its retries falls back to the GitHub copy in `models/`, which serves byte-identical files:

```sh
PPU_INDRA_OCR_MODEL_MIRROR=1 bun run your-script.ts
```

It is off by default. GitHub LFS bandwidth is a monthly budget, so turn it on where request volume is bounded and known. A model URL you pass yourself is never rewritten.

## Benchmark

Benches use a small zero-dependency harness (`bench/harness.ts`): in-process timing, round-robin scheduling across rounds so thermal/GC drift hits every task equally, reporting the median plus min/max/stddev. Run `bun task bench`.

Representative results on Apple M1 / Bun 1.4.2 (9 rounds) at the shipped defaults, `convert()` per page, opencv and canvas-native:

```bash
task                       median      +/-stddev        min        max
--------------------------------------------------------------------
opencv two-column           290.3 ms      30.6 ms   279.8 ms   374.9 ms
opencv table                243.5 ms       9.7 ms   236.9 ms   269.1 ms
opencv receipt              155.5 ms      18.0 ms   153.6 ms   209.4 ms
canvas-native two-column    590.9 ms      29.7 ms   554.6 ms   644.4 ms
canvas-native table         498.8 ms      17.6 ms   493.0 ms   537.7 ms
canvas-native receipt       327.7 ms      24.2 ms   319.6 ms   394.8 ms
```

`two-column` and `table` are 1700 x 2200 test pages (US letter at 200 dpi); `receipt` is `assets/receipt.jpg`, a 720 x 1280 phone photo. Times include orientation, deskew, detection, recognition, reading order and markdown.

### Against ppu-paddle-ocr

`bench/vs-paddle.ts` runs both libraries with their default options, each alone in its own process, and scores the output against transcripts. Apple M1 / Bun 1.4.2, ppu-paddle-ocr 6.6.1. Three accuracy measures, because they disagree for good reasons: **char-F1** counts the characters read and ignores order and spacing; **token-F1** counts whole words, so a dropped space costs; **CER** ignores case and spaces but counts order.

**Document pages** (`bun bench/vs-paddle.ts`: the test pages plus the receipt, 5 rounds each):

| Page            | ms Indra / Paddle | char-F1       | token-F1      | CER           |
| :-------------- | :---------------- | :------------ | :------------ | :------------ |
| two-column.png  | 304 / 503         | 1.000 / 1.000 | 1.000 / 0.972 | 0.001 / 0.683 |
| rotated-90.png  | 294 / 386         | 1.000 / 0.000 | 1.000 / 0.000 | 0.001 / 1.000 |
| rotated-180.png | 306 / 513         | 1.000 / 0.579 | 1.000 / 0.016 | 0.001 / 0.784 |
| table.png       | 244 / 267         | 1.000 / 1.000 | 1.000 / 1.000 | 0.007 / 0.007 |
| chinese.png     | 270 / 260         | 1.000 / 1.000 | 1.000 / 1.000 | 0.000 / 0.000 |
| receipt.jpg     | 153 / 117         | 0.979 / 0.997 | 0.873 / 0.954 | 0.024 / 0.003 |

Paddle reads across both columns on one line (CER 0.683 on two-column) and does not turn pages upright by default. On the real receipt photo, Paddle reads a few characters Indra misses.

**SROIE receipts** (`SROIE_DIR=... bun bench/vs-paddle.ts`: 50 receipts, every 12th of the ICDAR 2019 SROIE set):

| Library | median ms | p90 ms | peak RSS | char-F1 | token-F1 | CER   |
| :------ | :-------- | :----- | :------- | :------ | :------- | :---- |
| Indra   | 188       | 822    | 1162 MB  | 0.973   | 0.848    | 0.156 |
| Paddle  | 220       | 434    | 1368 MB  | 0.972   | 0.753    | 0.123 |

Both read the characters equally well. Indra keeps the spaces between words (token-F1 better on 40 of 50 receipts, worse on 7); Paddle's lower CER comes from line order, because SROIE transcripts move values away from their labels. Indra's slow tail is the 35-megapixel scans in the set, where shrinking the full image dominates.

## How the Port Is Checked

`scripts/golden/` renders test pages and records every stage of the Python reference (`indra-ocr` 0.3.1) on them: orientation, skew, detection, recognition, reading order, the text layer and the final page. The tests compare this package with those goldens. On the opencv engine the line text, blocks and markdown match exactly, and the line boxes agree to an IoU of 0.98 or better. The canvas-native engine stays within a token-F1 of 0.98.

## Ecosystem

ppu-indra-ocr is part of a family of document-processing libraries for JavaScript runtimes, all from [PT Perkasa Pilar Utama](https://github.com/PT-Perkasa-Pilar-Utama):

| Library                                                                                      | What it does                                                                                                    |
| :------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| [ppu-paddle-ocr](https://github.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr)                   | PaddleOCR (PP-OCRv6) text detection and recognition in every JavaScript runtime; the scaffold of this library.  |
| [ppu-ocv](https://github.com/PT-Perkasa-Pilar-Utama/ppu-ocv)                                 | Chainable image processing on OpenCV.js, plus canvas utilities that run in Node, Bun, browsers, and extensions. |
| [ppu-pdf](https://github.com/PT-Perkasa-Pilar-Utama/ppu-pdf)                                 | PDF text extraction (digital and scanned) with coordinates, line grouping, and page-to-canvas/PNG rendering.    |
| [ppu-doclayout](https://github.com/PT-Perkasa-Pilar-Utama/ppu-doclayout)                     | Document layout analysis with PaddlePaddle PP-DocLayout (tables, figures, text regions).                        |
| [ppu-doc-correction](https://github.com/PT-Perkasa-Pilar-Utama/ppu-doc-correction)           | Document image correction: page orientation, geometric unwarping (UVDoc), and text-line orientation.            |
| [ppu-uniface](https://github.com/PT-Perkasa-Pilar-Utama/ppu-uniface)                         | Face detection, recognition, verification, alignment, and anti-spoofing (a port of Python's Uniface).           |
| [ppu-yolo-onnx-inference](https://github.com/PT-Perkasa-Pilar-Utama/ppu-yolo-onnx-inference) | YOLOv11 object detection in Bun/Node and browsers; no Python or PyTorch required.                               |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code-quality requirements, and the pull request process.

## License

- **Code:** Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The pipeline is a TypeScript port of [datalab-to/indra](https://github.com/datalab-to/indra) (Apache-2.0). The library scaffold comes from [ppu-paddle-ocr](https://github.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr).
- **Model weights:** Datalab, modified OpenRAIL-M, see [models/LICENSE](models/LICENSE) and [models/README.md](models/README.md). The license's use restrictions apply to you and to anyone you pass the weights or a service on them to.
- **License status:** [LICENSE-STATUS.md](LICENSE-STATUS.md) records our request to Datalab and what may change. It is subject to change.
- Indra OCR is made by [Datalab](https://www.datalab.to). For commercial use beyond the license, or the highest accuracy, see their platform.

## Support

[Open an issue](https://github.com/PT-Perkasa-Pilar-Utama/ppu-indra-ocr/issues).

## Scripts

Recommended development environment is Linux-based. Library template: https://github.com/aquapi/lib-template

| Script                     | Command                              | Description                                                |
| :------------------------- | :----------------------------------- | :--------------------------------------------------------- |
| `bun task build`           | `bun run scripts/build.ts`           | Emit `.js` and `.d.ts` to `lib/`.                          |
| `bun task prepare-publish` | `bun run scripts/prepare-publish.ts` | Stage `package.json`, `README.md` and notices into `lib/`. |
| `bun task bench`           | `bun run scripts/bench.ts`           | Run `*.bench.ts` files.                                    |
| `bun bench/vs-paddle.ts`   | -                                    | Compare with ppu-paddle-ocr on document pages or SROIE.    |
| `bun run build:pages`      | build + `playground/build-pages.ts`  | Assemble the playground for Cloudflare.                    |

To run a specific benchmark file:

```bash
bun task bench index     # Run bench/index.bench.ts
```
