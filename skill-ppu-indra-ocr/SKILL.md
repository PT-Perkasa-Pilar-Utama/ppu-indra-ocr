---
name: skill-ppu-indra-ocr
description: Use this skill whenever the user writes TypeScript/JavaScript code that imports `ppu-indra-ocr`, `ppu-indra-ocr/web`, `ppu-indra-ocr/mobile` or `ppu-indra-ocr/pdf`, or asks about running Datalab's Indra OCR in a JavaScript runtime - image or PDF to markdown, text-line detection, line recognition, reading order and layout blocks, math as LaTeX, PDF text layers, ONNX Runtime OCR in Node, Bun, browsers or React Native. Trigger even when the user says "Indra OCR in Node" or "PDF to markdown in Bun" without naming the package, as long as the codebase imports it. This skill covers the entry points, the service lifecycle (`new` -> `initialize` -> `convert` -> `destroy`), the Page result, running single stages (orient, straighten, detect, recognize, order, readMath), PDF input through pdfium wasm, batches, the processing engines, the model cache and model license, the CLI, and the `apps/serve` REST server.
---

# Writing code with `ppu-indra-ocr`

`ppu-indra-ocr` is a TypeScript port of Datalab's Indra OCR pipeline on ONNX
Runtime. It turns an image or a PDF into markdown, plain text, line boxes and
layout blocks.

Read [references/configuration.md](references/configuration.md) for every
option. Read [references/recipes.md](references/recipes.md) for task-shaped
examples.

## The model license comes first

The code is Apache-2.0. The model weights are Datalab's, under a modified
OpenRAIL-M license. That license bans use by organizations above a revenue or
funding limit (except personal or research use) and by competitors of
Datalab. A hosted service on the weights counts as distribution. Point the
user to `models/README.md` when they plan commercial use. Do not tell them it
is fine.

## The entry points

| Import                 | Runtime                    | ONNX Runtime peer          | Default engine    |
| ---------------------- | -------------------------- | -------------------------- | ----------------- |
| `ppu-indra-ocr`        | Node.js, Bun               | `onnxruntime-node`         | `"opencv"`        |
| `ppu-indra-ocr/web`    | Browsers, web workers      | `onnxruntime-web`          | `"canvas-native"` |
| `ppu-indra-ocr/mobile` | React Native (iOS/Android) | `onnxruntime-react-native` | `"canvas-native"` |
| `ppu-indra-ocr/pdf`    | Any of the above           | needs `@embedpdf/pdfium`   | -                 |

Each runtime entry exports `IndraOcrService`, the one-shot `ocr()` and
`detect()`, `isPdf()`, the shared types, and the plain helpers `toMarkdown`,
`joinLines`, `orderXYCut` and `BLOCK_CLASSES`.

Status: Node and Bun are tested. The web entry has automated tests only in
Bun with a canvas polyfill; the playground was checked by hand in headless
Chromium. The mobile entry is type-checked and has not run on a device.
Say so when the user targets web or mobile.

## The lifecycle

```ts
import { IndraOcrService } from "ppu-indra-ocr";

const service = new IndraOcrService(); // options: see configuration.md
await service.initialize(); // loads (and on first run downloads) the models
const page = await service.convert(await Bun.file("scan.png").arrayBuffer());
console.log(page.markdown);
await service.destroy(); // releases the ONNX sessions
```

- Create one service per process and reuse it. `initialize()` is the slow
  part.
- The stage methods throw before `initialize()`. `isInitialized()` tells.
- `destroy()` releases the models. `initialize()` can load them again.
- For one image in a script, `ocr(image, options?)` and
  `detect(image, options?)` load the models, run, and release them.

## Inputs

`ImageInput` is one of:

- encoded bytes (`ArrayBuffer` or `Uint8Array`): PNG, JPEG, WebP, and what the
  platform canvas decodes,
- a canvas of the platform,
- gray pixels `{ data: Uint8Array, width, height }` (the `Gray` type).

PDFs do not go to `convert`. Route them with `isPdf(bytes)` and send them to
`convertPdf` from `ppu-indra-ocr/pdf`.

## The Page result

```ts
type Page = {
  markdown: string | null; // headings, paragraphs, lists; tables as HTML
  text: string; // every line, one per row, in reading order
  lines: { box: [x0, y0, x1, y1]; text: string; conf: number }[];
  blocks: { label: BlockLabel; bbox: [x0, y0, x1, y1]; text: string }[];
  source: "ocr" | "text" | "hybrid" | "error";
  note: string; // why the page took its path
  rotation: number; // degrees counter-clockwise the input was turned
  skew: number; // degrees of small tilt undone
  timings: Record<string, number>; // ms per stage, with total_ms
};
```

- Boxes are in the pixels of the upright page, after any quarter turn.
- `BlockLabel`: `Text`, `Section-Header`, `Page-Header`, `Page-Footer`,
  `List-Group`, `Table`, `Caption`, `Footnote`, `Table-Of-Contents`, `Form`.
- Page headers and footers are left out of `markdown` unless the service has
  `furniture: true`. They stay in `lines` and `blocks`.
- `source` is `text` or `hybrid` only for PDF pages that use the text layer.
  `error` means the page failed; `note` holds the reason.

## Running the stages yourself

The service exposes each stage, so the user can chain their own pipeline:

```ts
import { IndraOcrService, toMarkdown } from "ppu-indra-ocr";

const { turns } = await service.orient(image); // { turns, probs }
const { image: upright, skew } = await service.straighten(image, turns);
const det = await service.detect(upright); // [x0, y0, x1, y1, score][]
const boxes = det.map(([x0, y0, x1, y1]) => [x0, y0, x1, y1] as const);
const lines = await service.recognize(upright, boxes); // { text, conf, toks }[]
const texts = lines.map((l) => l.text);
const { order, cls, same } = await service.order(boxes, texts, upright.width, upright.height);
const markdown = cls && same ? toMarkdown(boxes, texts, order, cls, same) : null;
```

- `readMath(image, boxes)` reads boxes as text with inline LaTeX. It throws
  unless the service has `math: true`.
- `toGray(image)` decodes any `ImageInput` to `Gray`. Decode once when you run
  several stages on one image.
- `service.pipeline` holds lower-level steps (line splitting, assembly). Use
  it only when the stage methods are not enough.
- `convert` does more than this chain: an adaptive second detection pass,
  line splitting, script routing to the CJK recognizer, and math. Prefer
  `convert` unless the user needs a single stage.

## PDF input

```ts
import { convertPdf, isPdf } from "ppu-indra-ocr/pdf";

const pages = isPdf(bytes) ? await convertPdf(service, bytes) : [await service.convert(bytes)];
```

- Needs the optional peer `@embedpdf/pdfium` (pdfium as WebAssembly).
- `mode: "auto"` (default) uses the text layer where it passes the checks and
  OCR elsewhere. `"ocr"` always renders and OCRs. `"text"` trusts the layer.
- A page that fails returns `source: "error"`. A file that is not a PDF, is
  corrupt, or is locked without `password` throws.
- In a browser, pdfium's wasm comes from jsDelivr unless the user passes
  `pdfium: await loadPdfium(wasmBytesOrUrl)`.

## Batches

```ts
const pages = await service.convertBatch(images); // Page[] in input order
const settled = await service.convertBatch(images, { settle: true }); // never rejects per item
for await (const r of service.convertStream(images)) console.log(r.index, r.status);
```

`concurrency` defaults to 1. Each page allocates large tensors, so more pages
in flight mostly add memory. Do not raise it without a measurement. `signal`
aborts, `onProgress(done, total)` reports progress.

## Processing engines

| Engine            | Image operations            | Results                         |
| ----------------- | --------------------------- | ------------------------------- |
| `"opencv"`        | OpenCV.js through `ppu-ocv` | Match the Python reference      |
| `"canvas-native"` | Plain TypeScript, no OpenCV | Within token-F1 0.98 of the ref |

Pick `"opencv"` when results must match the Python `indra-ocr`. Pick
`"canvas-native"` for a smaller bundle or where OpenCV does not load.
On the web, `"opencv"` needs OpenCV.js on `globalThis.cv` first: a
`<script>` tag for `@techstark/opencv-js` or `import "@techstark/opencv-js"`.
The library does not fetch it. Mobile has no `"opencv"`.

## Models and the cache

- Models come from `datalab-to/indra-ocr` on Hugging Face, pinned to
  `MODEL_REVISION`. About 18 MB, 38 MB with math.
- Node caches them in `~/.cache/ppu-indra-ocr`. Warm the cache with
  `IndraOcrService.downloadModels({ math })` or `ppu-indra-ocr download-models`.
- `PPU_INDRA_OCR_MODEL_MIRROR=1` falls back to the GitHub mirror when Hugging
  Face fails.
- `models: { det: "/path/det.onnx", ... }` reads files from paths, URLs or
  `ArrayBuffer`s. Files left out use the default.
- The OCR reads Latin (about 50 languages), Cyrillic, Greek, Devanagari,
  Bengali and Chinese. There is no language option: a script model routes
  Chinese lines to the CJK recognizer.

## Common mistakes to flag

- A new service per request. Create one at startup and reuse it.
- `concurrency: 4` or more "for speed". It mostly adds memory.
- PDF bytes passed to `convert`. Use `convertPdf`.
- `readMath` on a service without `math: true`.
- Expecting Arabic, Hebrew, Thai, Japanese kana, Korean or handwriting from
  the OCR. The models do not read them. A PDF text layer in those scripts
  works.
- Ignoring `page.source === "error"` in PDF results.
- Commercial use without reading the weights license.

## The CLI and the REST server

For a shell task, use the CLI instead of a script:

```bash
ppu-indra-ocr convert paper.pdf                     # markdown to stdout
ppu-indra-ocr convert "scans/*.png" --out results/  # one file per input
ppu-indra-ocr convert scan.png --format json --pretty
ppu-indra-ocr detect receipt.jpg                    # line boxes as JSON
```

For OCR as a service, use `apps/serve` (Hono + Bun): `POST /v1/ocr` takes an
image or a PDF and returns `{ pages }`. Its env vars are in
[references/configuration.md](references/configuration.md).
