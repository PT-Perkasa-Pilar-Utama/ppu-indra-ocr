# Configuration reference

Every option of `IndraOcrService`, `convertBatch`, `convertPdf`, and the
`apps/serve` environment. Every field is optional.

## `IndraOptions` (service constructor)

```ts
new IndraOcrService({
  model?: "v1.1" | "v3.1";
  models?: ModelSources;
  lm?: boolean;
  math?: boolean;
  furniture?: boolean;
  adaptive?: boolean;
  engine?: "opencv" | "canvas-native";
  session?: SessionOptions;
  verbose?: boolean;
});
```

| Option      | Default                              | Meaning                                          |
| ----------- | ------------------------------------ | ------------------------------------------------ |
| `model`     | `"v1.1"`                             | Model preset: Indra OCR 1.1, or `"v3.1"` (1.0)   |
| `models`    | Datalab's repo, pinned               | Per-file sources, see below                      |
| `lm`        | Node: `true`, else `false`           | Re-read unsure lines with the language models    |
| `math`      | `false`                              | Read math lines as LaTeX. Loads four more models |
| `furniture` | `false`                              | Keep page headers and footers in `markdown`      |
| `adaptive`  | `true`                               | Second detection pass on pages with small text   |
| `engine`    | Node: `opencv`, else `canvas-native` | Image engine. Mobile has no `opencv`             |
| `session`   | CPU, CPU memory arena off            | ONNX Runtime session options for every model     |
| `verbose`   | `false`                              | Log loading and stage steps to the console       |

`ocr(image, options)` and `detect(image, options)` take the same options.

## `models` (`ModelSources`)

A partial map from file role to a source: a URL, a file path (Node only), or
an `ArrayBuffer`. Roles left out use `defaultModelUrl(role, preset)`; a role
the preset lacks (v3.1 has no `recRtl`, `recWorld` or `lm*`) stays off.

| Role          | File                 | Loaded                                            |
| ------------- | -------------------- | ------------------------------------------------- |
| `det`         | `det.onnx`           | always                                            |
| `rec`         | `rec.onnx`           | always                                            |
| `script`      | `script.onnx`        | always                                            |
| `orient`      | `orient.onnx`        | always                                            |
| `order`       | `order.onnx`         | always                                            |
| `meta`        | `meta.json`          | always (vocabulary)                               |
| `recCjk`      | `rec_cjk.onnx`       | first CJK line                                    |
| `recRtl`      | `rec_rtl.onnx`       | first RTL line (v1.1)                             |
| `recWorld`    | `rec_world.onnx`     | first Indic, Thai, ... line (v1.1)                |
| `lm`          | `lm.npz`             | first unsure line, `lm: true` (v1.1)              |
| `lmCjk`       | `lm_cjk.npz`         | first unsure CJK line, `lm: true` (v1.1)          |
| `lmWorld`     | `lm_world.npz`       | first unsure other-script line, `lm: true` (v1.1) |
| `mathCls`     | `mathcls.onnx`       | `math: true`                                      |
| `mathEnc`     | `math_enc.onnx`      | `math: true`                                      |
| `mathDecInit` | `math_dec_init.onnx` | `math: true`                                      |
| `mathDecStep` | `math_dec_step.onnx` | `math: true`                                      |
| `mathMeta`    | `math_meta.json`     | `math: true`                                      |

Exported constants: `MODEL_FILES` maps role to file name, `MATH_FILES` lists
the math roles, `MODEL_PRESETS` holds each preset's commit, tag, URLs and
files, `presetFiles(preset, { math, lm })` lists what a service can load,
`MODEL_REVISION` and `MODEL_BASE_URL` name the default preset, and
`MODEL_MIRROR_BASE_URL` is the GitHub mirror root (`v1.1/` for v1.1).

Use a different model revision only with a reason. The goldens and the
vocabulary in `meta.json` are tied to the presets' commits.

## `session` (`SessionOptions`)

ONNX Runtime's `InferenceSession.SessionOptions`, plus one hook:

- `executionProviders`: for example `["cuda", "cpu"]` on Node, `["wasm"]` on
  web. Default CPU (wasm on web).
- `onSessionFallback(error)`: called when the requested providers fail and the
  service retries on `cpu` or `wasm`. Without it the fallback is silent.

The service sets `enableCpuMemArena: false` unless the user passes it. The
arena keeps peak memory after each page; turning it on trades memory for a
little speed.

WebGPU is not picked by default and is untested with these models.

## `BatchOptions` (`convertBatch`, `convertStream`)

| Option        | Default | Meaning                                                           |
| ------------- | ------- | ----------------------------------------------------------------- |
| `concurrency` | `1`     | Pages in flight at once                                           |
| `settle`      | `false` | Return `{ index, status, value \| reason }` per image             |
| `signal`      | -       | `AbortSignal`: stops scheduling, rejects with `AbortError`        |
| `onProgress`  | -       | `(done, total)` after each image; `total` undefined for iterables |

`convertBatch` keeps input order. `convertStream` yields in completion order
with `index` set.

## `ConvertPdfOptions` (`convertPdf`)

| Option     | Default  | Meaning                                                                                   |
| ---------- | -------- | ----------------------------------------------------------------------------------------- |
| `mode`     | `"auto"` | `auto`: text layer where it passes the checks. `ocr`: always OCR. `text`: trust the layer |
| `dpi`      | `200`    | Render resolution for OCR                                                                 |
| `imageOcr` | `true`   | On text-layer pages, OCR lines with no layer characters (text in images)                  |
| `maxPages` | all      | Read at most this many pages (`0`: all)                                                   |
| `password` | -        | Password of an encrypted PDF                                                              |
| `pdfium`   | shared   | A module from `loadPdfium(wasm?)`; loaded once and reused when left out                   |

`PDFIUM_VERSION` names the `@embedpdf/pdfium` release the package is tested
with. A browser fetches that version's wasm from jsDelivr by default.

## CLI flags

`ppu-indra-ocr convert <files...>`: `--format md|text|json`, `--pretty`,
`--out <dir>`, `--mode auto|ocr|text`, `--dpi`, `--max-pages`, `--password`,
`--math`, `--keep-furniture`, `--engine`, `--no-image-ocr`, `--model v1.1|v3.1`,
`--no-lm`, `--model-dir`, `--execution-providers`, `--timings`, `--quiet`, `--verbose`. Run `ppu-indra-ocr help`
for the current list.

## `apps/serve` environment

| Variable                    | Default            | Meaning                                       |
| --------------------------- | ------------------ | --------------------------------------------- |
| `PORT` / `HOST`             | `8080` / `0.0.0.0` |                                               |
| `SECRET_KEY`                | -                  | If set, `Bearer <key>` is required on `/v1/*` |
| `MAX_UPLOAD_BYTES`          | `10485760`         | Per image or PDF                              |
| `MAX_BATCH_IMAGES`          | `32`               |                                               |
| `EXECUTION_PROVIDERS`       | `cpu`              | Comma list, for example `cuda,cpu`            |
| `DEFAULT_ENGINE`            | `opencv`           | `opencv` or `canvas-native`                   |
| `MODEL`                     | `v1.1`             | Model preset: `v1.1` or `v3.1`                |
| `MODEL_DIR`                 | default download   | A folder with the Indra model files           |
| `LM`                        | `true`             | Re-read unsure lines with the language models |
| `MATH`                      | `false`            | Read math as LaTeX                            |
| `KEEP_FURNITURE`            | `false`            | Keep page headers and footers in the markdown |
| `PDF_DPI` / `MAX_PDF_PAGES` | `200` / `50`       | PDF render resolution; pages read per PDF     |
| `MAX_CONCURRENCY`           | `0` (auto = 1)     | Inferences in flight                          |
| `MAX_QUEUE_DEPTH`           | `100`              | Past this, `429` with `Retry-After`           |

`apps/serve/README.md` lists the rest (rate limits, CORS, IP lists, docs,
timeouts, async task retention, source URL allowlist).
