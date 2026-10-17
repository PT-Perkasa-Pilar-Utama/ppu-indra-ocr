# ppu-indra-ocr

TypeScript SDK for [Indra OCR](https://github.com/datalab-to/indra) by
Datalab. It reads images and PDFs into markdown, plain text, or JSON with
line boxes and layout blocks, on ONNX Runtime.

> **Status: early development (0.x).** The API may change between minor versions.

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

## Table of contents

- [What it does](#what-it-does)
- [Runtime support](#runtime-support)
- [Installation](#installation)
- [Usage](#usage)
- [The page result](#the-page-result)
- [Running the stages yourself](#running-the-stages-yourself)
- [PDF input](#pdf-input)
- [Batches](#batches)
- [Math](#math)
- [Processing engine](#processing-engine)
- [Web and browser](#web-and-browser)
- [React Native](#react-native)
- [Command line](#command-line)
- [REST server](#rest-server)
- [Models and languages](#models-and-languages)
- [Configuration reference](#configuration-reference)
- [Limitations](#limitations)
- [How the port is checked](#how-the-port-is-checked)
- [License and credits](#license-and-credits)

## What it does

Indra OCR is a small CPU OCR pipeline: about 18 MB of ONNX models, with a
1M-parameter line recognizer. This package runs the whole pipeline in
TypeScript:

```
orient.onnx (page upright?) -> skew fix -> det.onnx (text lines)
  -> script.onnx (Chinese line?) -> rec.onnx | rec_cjk.onnx (text)
  -> order.onnx (reading order, layout labels, blocks) -> markdown
  [math: true]  mathcls -> math encoder -> decoder (LaTeX)
  [PDF]         the PDF's text layer where it can be trusted, OCR elsewhere
```

Each stage is also a method, so you can run the whole page with one call or
chain the stages yourself.

## Runtime support

| Runtime                    | Entry                  | ONNX Runtime               | Status                                 |
| -------------------------- | ---------------------- | -------------------------- | -------------------------------------- |
| Node.js, Bun               | `ppu-indra-ocr`        | `onnxruntime-node`         | Tested                                 |
| Browsers, web workers      | `ppu-indra-ocr/web`    | `onnxruntime-web` (wasm)   | Tested in Bun; run by hand in Chromium |
| React Native (iOS/Android) | `ppu-indra-ocr/mobile` | `onnxruntime-react-native` | Type-checked, not run on a device yet  |
| PDF input (any of above)   | `ppu-indra-ocr/pdf`    | -                          | Tested in Bun                          |

## Installation

```bash
bun add ppu-indra-ocr onnxruntime-node
# PDF input (optional):
bun add @embedpdf/pdfium
```

ONNX Runtime and pdfium are peer dependencies: install the ones your runtime
needs (`onnxruntime-web` in a browser, `onnxruntime-react-native` and
`@shopify/react-native-skia` on React Native).

The first `initialize()` downloads the models (about 18 MB, 38 MB with math)
from Datalab's Hugging Face repository, pinned to one revision, and caches
them in `~/.cache/ppu-indra-ocr` on Node.

**When the models download, you accept Datalab's license for the weights.**
Check [LICENSE-STATUS.md](LICENSE-STATUS.md) first.

## Usage

```ts
import { IndraOcrService } from "ppu-indra-ocr";

const service = new IndraOcrService();
await service.initialize();

const page = await service.convert(await Bun.file("scan.png").arrayBuffer());
console.log(page.markdown);

await service.destroy();
```

`convert` takes encoded image bytes (`ArrayBuffer` or `Uint8Array`: PNG, JPEG,
WebP and the other formats your canvas decodes), a canvas, or gray pixels
`{ data: Uint8Array, width, height }`.

For one image, the one-shot functions load the models, run, and release them:

```ts
import { detect, ocr } from "ppu-indra-ocr";

const page = await ocr(imageBytes);
const boxes = await detect(imageBytes); // [x0, y0, x1, y1, score][]
```

## The page result

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

Block labels: `Text`, `Section-Header`, `Page-Header`, `Page-Footer`,
`List-Group`, `Table`, `Caption`, `Footnote`, `Table-Of-Contents`, `Form`.
Page headers and footers are left out of the markdown unless you pass
`furniture: true`. Boxes are in the upright page's pixels.

## Running the stages yourself

```ts
const { turns } = await service.orient(image); // quarter turns, from orient.onnx
const { image: upright, skew } = await service.straighten(image, turns);
const boxes = (await service.detect(upright)).map(([x0, y0, x1, y1]) => [x0, y0, x1, y1] as const);
const lines = await service.recognize(upright, boxes); // text, confidence, character positions
const texts = lines.map((l) => l.text);
const { order, cls, same } = await service.order(boxes, texts, upright.width, upright.height);
```

`recognize` takes boxes from any detector, one line per box. Boxes should hug
the text: the recognizer adds 0.2 x the box height above and below and 0.3 x
on each side. Trim boxes that carry their own padding first. PaddleOCR's
boxes, for example, are padded to 1.8 x the text height.

`toMarkdown`, `joinLines` and `orderXYCut` are exported as plain functions.
`service.pipeline` gives the lower-level stages (line splitting, assembly)
when you need them.

## PDF input

```ts
import { IndraOcrService } from "ppu-indra-ocr";
import { convertPdf } from "ppu-indra-ocr/pdf";

const service = new IndraOcrService();
await service.initialize();
const pages = await convertPdf(service, pdfBytes, { mode: "auto" });
```

| Option     | Default  | Meaning                                                                                                                               |
| ---------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`     | `"auto"` | `auto`: the text layer where it passes the checks, OCR elsewhere. `ocr`: always OCR. `text`: the text layer even when the checks fail |
| `dpi`      | `200`    | Render resolution for OCR                                                                                                             |
| `pdfOrder` | `true`   | Keep the PDF content order when it is coherent                                                                                        |
| `imageOcr` | `true`   | On text-layer pages, OCR lines with no text-layer characters (text in images)                                                         |
| `maxPages` | all      | Read at most this many pages                                                                                                          |
| `password` | -        | Password of an encrypted PDF                                                                                                          |

The text layer is checked first: invisible text (an OCR layer), broken font
maps, and runs with no ink under them send the page to OCR. A page that fails
comes back with `source: "error"` and the reason in `note`; a file that cannot
be opened throws. pdfium runs as WebAssembly (`@embedpdf/pdfium`); a browser
fetches its wasm from jsDelivr unless you pass `loadPdfium(wasmBytesOrUrl)`.
`isPdf(bytes)` is exported from every entry to route files without loading
pdfium.

## Batches

```ts
const pages = await service.convertBatch(images, { concurrency: 1 });
const settled = await service.convertBatch(images, { settle: true }); // { index, status, value | reason }[]

for await (const r of service.convertStream(images)) console.log(r.index, r.status);
```

`concurrency` defaults to 1: each page allocates large tensors, so more pages
at once mostly add memory. `signal` aborts, `onProgress` reports each image.

## Math

```ts
const service = new IndraOcrService({ math: true });
```

A classifier finds lines that hold math, and an attention decoder reads them
as text with inline LaTeX (`we have $x _ { i } \leq 1$`). A reading replaces a
line only when it holds math and its mean token probability is 0.9 or more.
Math loads four more models and is slower on pages that have it.

## Processing engine

| Engine            | Image operations              | Results                               | Default on  |
| ----------------- | ----------------------------- | ------------------------------------- | ----------- |
| `"opencv"`        | OpenCV.js (through `ppu-ocv`) | Match the Python reference            | Node, Bun   |
| `"canvas-native"` | Plain TypeScript, no OpenCV   | Within token-F1 0.98 of the reference | Web, mobile |

The opencv engine runs the same OpenCV release (5.0) as the reference. The
canvas-native engine is smaller and runs anywhere, with slightly different
resampling.

## Web and browser

```ts
import { IndraOcrService } from "ppu-indra-ocr/web";

const service = new IndraOcrService(); // canvas-native, wasm
await service.initialize();
const page = await service.convert(await file.arrayBuffer());
```

Models are fetched from the default URLs or from `models`. ONNX Runtime runs
on wasm unless `session.executionProviders` asks for more; WebGPU is untested
with these models.

`engine: "opencv"` matches the reference exactly, but the page must load
OpenCV.js first. The library does not fetch it (about 10 MB):

```html
<script src="https://cdn.jsdelivr.net/npm/@techstark/opencv-js@5.0.0-release.1/dist/opencv.js"></script>
```

Only the script tag is tested. Without OpenCV.js on `globalThis.cv`,
`initialize()` throws and says so.

The automated web tests run in Bun with a canvas polyfill. The
[playground](playground) runs the web and PDF entries in a real browser; it
was checked by hand in headless Chromium (images, PDFs, both engines).

## React Native

```ts
import { IndraOcrService } from "ppu-indra-ocr/mobile";
```

Uses `onnxruntime-react-native` and Skia (through `ppu-ocv/canvas-mobile`)
with the canvas-native engine. This entry is type-checked but has not run on
a device yet.

## Command line

```bash
ppu-indra-ocr convert paper.pdf                     # markdown to stdout
ppu-indra-ocr convert "scans/*.png" --out results/  # one file per input
ppu-indra-ocr convert scan.png --format json --pretty
ppu-indra-ocr convert paper.pdf --mode ocr --math --timings
ppu-indra-ocr detect receipt.jpg                    # line boxes as JSON
ppu-indra-ocr download-models                       # also: clear-cache, models
```

Run `ppu-indra-ocr help` for every flag (`--keep-furniture`, `--engine`,
`--dpi`, `--max-pages`, `--password`, `--model-order`, `--no-image-ocr`,
`--model-dir`, `--execution-providers`).

## REST server

[apps/serve](apps/serve) wraps the service as an HTTP API (Hono + Bun):
`POST /v1/ocr` for an image or a PDF, `/v1/detect`, batch, stream and async
endpoints, a bounded inference queue, auth, metrics and OpenAPI docs. A hosted
service counts as distribution under the weights' license; see its README.

## Models and languages

The models come from
[datalab-to/indra-ocr](https://huggingface.co/datalab-to/indra-ocr), pinned to
commit `d7c839c` (tag v3.1). `models/` in this repository holds an unmodified
mirror, checked file by file against upstream. Set
`PPU_INDRA_OCR_MODEL_MIRROR=1` to fall back to it when Hugging Face fails.
Pass `models` to read some or all files from paths, URLs or bytes.

- **OCR (scans and images):** Latin (about 50 languages), Cyrillic, Greek,
  Devanagari, Bengali, and Chinese (simplified and traditional).
- **PDF text layer:** any script.

## Configuration reference

| Option      | Default                   | Meaning                                                                                           |
| ----------- | ------------------------- | ------------------------------------------------------------------------------------------------- |
| `models`    | Datalab's repo, pinned    | Per-file sources: `det`, `rec`, `recCjk`, `script`, `orient`, `order`, `meta`, and the math files |
| `math`      | `false`                   | Read math lines as LaTeX                                                                          |
| `furniture` | `false`                   | Keep page headers and footers in the markdown                                                     |
| `adaptive`  | `true`                    | Second detection pass on pages with small text                                                    |
| `engine`    | `"opencv"` on Node        | `"opencv"` or `"canvas-native"`                                                                   |
| `session`   | CPU, CPU memory arena off | ONNX Runtime session options; `onSessionFallback` hook                                            |
| `verbose`   | `false`                   | Log loading steps                                                                                 |

## Limitations

From the upstream model card: Indra OCR is built for speed on clean print.
Complex tables (merged or spanning cells, no ruling lines), handwriting,
scripts outside the OCR list (Arabic, Hebrew, Thai, Japanese kana, Korean on
scans), degraded scans, vertical text and very wide pages read poorly.

In this port, the right-to-left check for PDF text-layer lines uses script
ranges instead of Unicode bidi classes.

## How the port is checked

`scripts/golden/` renders test pages and records every stage of the Python
reference (`indra-ocr` 0.3.1) on them: orientation, skew, detection,
recognition, reading order, the text layer and the final page. The tests
compare this package with those goldens. On the opencv engine the line text,
blocks and markdown match exactly, and the line boxes agree to an IoU of 0.98
or better. The canvas-native engine stays within a token-F1 of 0.98.

## License and credits

- **Code:** Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). The
  pipeline is a TypeScript port of
  [datalab-to/indra](https://github.com/datalab-to/indra) (Apache-2.0). The
  library scaffold comes from
  [ppu-paddle-ocr](https://github.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr).
- **Model weights:** Datalab, modified OpenRAIL-M, see
  [models/LICENSE](models/LICENSE) and [models/README.md](models/README.md).
  The license's use restrictions apply to you and to anyone you pass the
  weights or a service on them to.
- **License status:** [LICENSE-STATUS.md](LICENSE-STATUS.md) records our
  request to Datalab and what may change. It is subject to change.
- Indra OCR is made by [Datalab](https://www.datalab.to). For commercial use
  beyond the license, or the highest accuracy, see their platform.
