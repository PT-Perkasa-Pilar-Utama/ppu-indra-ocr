# `ppu-indra-ocr` recipes

Each recipe assumes the matching install: `ppu-indra-ocr` plus
`onnxruntime-node` on Node and Bun, or `onnxruntime-web` in a browser. PDF
recipes also need `@embedpdf/pdfium`.

## 1. Warm the model cache in CI or Docker

Download once at build time, so the first request does not wait on Hugging
Face:

```bash
ppu-indra-ocr download-models          # base models
ppu-indra-ocr download-models --math   # plus the math models
```

Or from code:

```ts
import { IndraOcrService } from "ppu-indra-ocr";

await IndraOcrService.downloadModels({ math: false });
```

Files land in `~/.cache/ppu-indra-ocr`. In CI, cache that folder with a key
on the package version, because the model revision is pinned per release. Set
`PPU_INDRA_OCR_MODEL_MIRROR=1` when shared runner IPs hit Hugging Face rate
limits.

## 2. Load models from a local folder

For air-gapped hosts, or to skip the download:

```ts
import { IndraOcrService, MODEL_FILES } from "ppu-indra-ocr";
import type { ModelSources } from "ppu-indra-ocr";
import path from "node:path";

const dir = "/opt/indra-models";
const models: ModelSources = Object.fromEntries(
  Object.entries(MODEL_FILES).map(([role, file]) => [role, path.join(dir, file)])
);
const service = new IndraOcrService({ models });
```

The folder must hold the files of the pinned revision. Copy them from the
cache or from `models/` in the repository.

## 3. One warm service in a long-running server

```ts
import { IndraOcrService } from "ppu-indra-ocr";
import { convertPdf, isPdf } from "ppu-indra-ocr/pdf";

const service = new IndraOcrService();
await service.initialize(); // once, at startup

export async function handle(bytes: Uint8Array) {
  return isPdf(bytes)
    ? convertPdf(service, bytes, { maxPages: 50 })
    : [await service.convert(bytes)];
}

process.on("SIGTERM", async () => {
  await service.destroy();
  process.exit(0);
});
```

Put a queue in front of `handle`. One page at a time is the default for a
reason: each page allocates large tensors. `apps/serve` already does this
(bounded queue, auth, limits); prefer it over a hand-rolled server.

## 4. Convert a folder of images on Bun

```ts
import { IndraOcrService } from "ppu-indra-ocr";
import { Glob } from "bun";

const service = new IndraOcrService();
await service.initialize();

const files = [...new Glob("scans/*.{png,jpg}").scanSync()];
async function* images() {
  for (const f of files) yield await Bun.file(f).arrayBuffer();
}
for await (const r of service.convertStream(images(), { settle: true })) {
  const out = files[r.index]?.replace(/\.\w+$/, ".md") ?? `page-${r.index}.md`;
  if (r.status === "fulfilled") await Bun.write(out, r.value.markdown ?? r.value.text);
  else console.error(files[r.index], r.reason);
}
await service.destroy();
```

An async generator reads each file only when the pool asks for it, so the
folder never sits in memory at once. `settle: true` keeps one bad file from
failing the run. The CLI does the same: `ppu-indra-ocr convert "scans/*.png" --out results/`.

## 5. Detection only

For line boxes without recognition (cropping, layout checks):

```ts
const boxes = await service.detect(image); // [x0, y0, x1, y1, score][]
```

`detect` runs on the image as given. Run `orient` and `straighten` first when
the input may be turned or tilted; the boxes are then in the straightened
image's pixels.

## 6. Re-read a region with your own boxes

```ts
const gray = await service.toGray(image); // decode once
const lines = await service.recognize(gray, [[120, 40, 480, 72]]);
console.log(lines[0]?.text, lines[0]?.conf);
```

`recognize` reads each box as one line. Give it line-height boxes; a box over
several lines reads poorly.

## 7. Math as LaTeX

```ts
const service = new IndraOcrService({ math: true });
await service.initialize();
const page = await service.convert(image); // math lines hold inline LaTeX
```

`convert` replaces a line with its math reading only when the line holds math
and the mean token probability is 0.9 or more. To read chosen boxes directly:
`await service.readMath(image, boxes)` returns `{ text, conf }` per box.

## 8. PDFs in the browser with self-hosted assets

```ts
import { IndraOcrService, MODEL_FILES } from "ppu-indra-ocr/web";
import type { ModelSources } from "ppu-indra-ocr/web";
import { convertPdf, loadPdfium } from "ppu-indra-ocr/pdf";

const models: ModelSources = Object.fromEntries(
  Object.entries(MODEL_FILES).map(([role, file]) => [role, `/models/${file}`])
);
const service = new IndraOcrService({ models });
await service.initialize();
const pdfium = await loadPdfium("/pdfium.wasm");
const pages = await convertPdf(service, new Uint8Array(await file.arrayBuffer()), { pdfium });
```

Copy `pdfium.wasm` from `@embedpdf/pdfium` at version `PDFIUM_VERSION`. The
web entry keeps no model cache of its own; the browser HTTP cache applies.
The playground runs this flow in Chromium with the default model and wasm
URLs. Self-hosted URLs are not tested.

## 9. Debugging a bad page

Check, in this order:

1. `page.rotation` and `page.skew`: a wrong turn ruins everything after it.
2. `page.note` and `page.source` on PDF pages: `text` with bad characters
   means a broken text layer; retry with `{ mode: "ocr" }`.
3. `service.detect(image)`: missing boxes mean small or faint text. Leave
   `adaptive` on, or render PDFs at a higher `dpi`.
4. `engine: "opencv"`: if the result changes, the canvas-native resampling is
   the cause.
5. `page.timings`: which stage is slow.

Then compare with the Python reference (`pip install indra-ocr`) on the same
image. If Python reads it wrong too, it is the model, not this package.
