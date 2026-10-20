# ppu-indra-ocr-serve

Production-grade REST API around [`ppu-indra-ocr`](../../README.md) - Hono + Bun. POST an image, get OCR JSON.

Pick the path that fits you:

### A) Run the published image (no clone)

For deploying or just trying it - nothing to build, models are pre-baked.

```bash
docker run -p 8080:8080 ghcr.io/pt-perkasa-pilar-utama/ppu-indra-ocr/serve:latest
curl -F file=@receipt.jpg http://localhost:8080/v1/ocr
```

Images publish to GitHub Container Registry on each release (slimmed with docker-slim). GPU users build the CUDA image from `Dockerfile.cuda`. Configure via `-e` env vars (see below).

### B) Clone & run from source

For developing or self-building. From the repo root:

```bash
cd apps/serve
bun install
cp .env.example .env          # optional - sane defaults otherwise
bun run dev                   # watch mode on http://localhost:8080
```

Or build the image yourself (from the **repo root**, the build context):

```bash
docker compose -f apps/serve/docker-compose.yml up --build
```

Open **http://localhost:8080/docs** for the Scalar API reference.

## Model license

The Indra OCR weights are Datalab's, under a modified OpenRAIL-M license
([models/LICENSE](../../models/LICENSE)). That license counts a hosted service
as distribution: give your users its use restrictions (Attachment A) and a
copy of the license, and check the revenue, funding and competitor limits
before you deploy.

## Why

The library is a building block; this wraps it as a service you'd be comfortable running in production: one warmed `IndraOcrService` shared behind a **bounded inference queue** (no OOM, no VRAM blow-up), graceful lifecycle, optional auth, Prometheus metrics, and OpenAPI docs.

## Endpoints

| Method | Path                      | Purpose                                                                           |
| ------ | ------------------------- | --------------------------------------------------------------------------------- |
| POST   | `/v1/ocr`                 | Sync OCR of an image or PDF - `multipart/form-data` (`file`) or JSON `{ source }` |
| POST   | `/v1/detect`              | Detection only - boxes, no recognition (same input as `/v1/ocr`)                  |
| POST   | `/v1/ocr/batch`           | Sync batch - JSON `{ sources: string[] }`                                         |
| POST   | `/v1/ocr/stream`          | SSE - one event per image as it finishes                                          |
| POST   | `/v1/ocr/async`           | Enqueue a batch -> `202 { taskId }`                                               |
| GET    | `/v1/tasks/:id`           | Task status                                                                       |
| GET    | `/v1/tasks/:id/result`    | Task result (409 until done)                                                      |
| DELETE | `/v1/tasks/:id`           | Cancel a task                                                                     |
| GET    | `/v1/models`              | Engines, model revision, defaults                                                 |
| GET    | `/health` / `/ready`      | Liveness / readiness (200 once warmed)                                            |
| GET    | `/metrics`                | Prometheus                                                                        |
| GET    | `/docs` / `/openapi.json` | Scalar UI / spec                                                                  |

### Input

`POST /v1/ocr` accepts `multipart/form-data` with a `file` field, or JSON:

```jsonc
{
  "source": "data:application/pdf;base64,...", // an image or a PDF
  "engine": "opencv", // opencv | canvas-native
  "mode": "auto", // PDFs: auto | ocr | text
  "maxPages": 10, // PDFs: at most this many pages (capped by MAX_PDF_PAGES)
}
```

A PDF reads its text layer where it can be trusted and OCRs the rest; an image is one page. The batch, stream, and async endpoints take images.

`source` must be a `data:` URI or an `https` URL whose host is in `SOURCE_URL_ALLOWLIST` (empty = https disabled). **Local filesystem paths are rejected**, and URL fetches refuse redirects - so the API never reads arbitrary host files or gets steered off-allowlist. Uploads are sniffed by magic bytes; anything that is not an image or a PDF gets a `400`, not a `500`. An allowlisted host that cannot be reached (DNS, TLS, refused, or a redirect) is a `502` naming the host.

`POST /v1/detect` takes an image (`file` or `{ source, engine? }`) and returns `{ boxes: [{ box: [x0, y0, x1, y1], score }] }` - line detection only, no recognition. `metadata` carries `speed`, `count`, and `engine`.

### Response format

Every JSON response uses a consistent envelope and carries the request id (also returned as the `X-Request-Id` header):

```jsonc
// success
{
  "status": "success",
  "version": "0.1.2",
  "metadata": { "id": "<request-id>", "speed": 0.27, "pages": 1, "engine": "opencv" },
  "data": { "pages": [{ "markdown": "...", "text": "...", "lines": [ ... ], "blocks": [ ... ], "source": "ocr", "rotation": 0, "skew": 0 }] }
}
// error
{ "status": "error", "version": "0.1.2", "data": { "message": "...", "requestId": "<request-id>" } }
```

`/metrics` is the only exception (Prometheus text). The spec at `/openapi.json` (rendered at `/docs`) is generated from the valibot schemas via `hono-openapi`.

## Configuration (env)

See [`.env.example`](.env.example) for the full annotated list.

| Var                                                   | Default            | Notes                                                               |
| ----------------------------------------------------- | ------------------ | ------------------------------------------------------------------- |
| `API_ENV`                                             | `development`      | `development` \| `production`                                       |
| `PORT` / `HOST`                                       | `8080` / `0.0.0.0` |                                                                     |
| `SECRET_KEY`                                          | -                  | If set, `Bearer <key>` required on `/v1/*` (`openssl rand -hex 32`) |
| `IP_WHITE_LIST` / `IP_DENY_LIST`                      | `*` / -            | Comma lists; `*` allows all (deny still applies)                    |
| `CORS_ORIGINS`                                        | `*`                | Comma list or `*`                                                   |
| `DOCS_ENABLED`                                        | `true`             | Serve `/docs` + `/openapi.json`                                     |
| `RATE_LIMIT_ENABLED`                                  | `true`             | Fixed-window per client IP on `/v1/*`                               |
| `RATE_LIMIT_PER_WINDOW` / `RATE_LIMIT_WINDOW_SECONDS` | `120` / `60`       | `429` + `Retry-After` past the limit                                |
| `REQUEST_TIMEOUT_SECONDS`                             | `30`               |                                                                     |
| `MAX_UPLOAD_BYTES`                                    | `10485760`         | Per image or PDF                                                    |
| `MAX_BATCH_IMAGES`                                    | `32`               |                                                                     |
| `EXECUTION_PROVIDERS`                                 | `cpu`              | Comma list, e.g. `cuda,cpu`                                         |
| `DEFAULT_ENGINE`                                      | `opencv`           | `opencv` \| `canvas-native`                                         |
| `MODEL`                                               | `v1.1`             | Model preset: `v1.1` (Indra OCR 1.1) \| `v3.1` (1.0, smaller)       |
| `MODEL_DIR`                                           | default download   | A folder with the Indra model files                                 |
| `LM`                                                  | `true`             | Re-read unsure lines with the language models (v1.1)                |
| `MATH`                                                | `false`            | Write math as LaTeX (loads 4 more models)                           |
| `KEEP_FURNITURE`                                      | `false`            | Keep page headers and footers in the markdown                       |
| `PDF_DPI` / `MAX_PDF_PAGES`                           | `200` / `50`       | PDF render resolution for OCR; pages read per PDF                   |
| `MAX_CONCURRENCY`                                     | `0` (auto)         | Auto = 1: each page allocates large tensors                         |
| `MAX_QUEUE_DEPTH`                                     | `100`              | Excess inferences get `429` + `Retry-After`                         |
| `TASK_TTL_SECONDS`                                    | `600`              | Async task retention                                                |
| `SOURCE_URL_ALLOWLIST`                                | -                  | Comma list of allowed https hosts                                   |

## Architecture

`bun test` runs HTTP-layer tests (no model load). The app imports the library from source via a tsconfig path (`../../src`), so no build step is needed in dev; it's a standalone package (its own `node_modules`), kept out of the library's workspace so the published package is unaffected.

Layout: shared infrastructure in `src/lib/` (config, queue, service, metrics, input, api-response), and one vertical slice per endpoint in `src/modules/<endpoint>/` (each a `createRoute` + handler registered on the `OpenAPIHono` app).

## Notes & limits

- **Concurrency model.** Single-image OCR flows through the shared queue. Batch endpoints use `convertBatch`'s own bounded concurrency (defaults to the same value); they aren't additionally gated by the single-image queue.
- **Async tasks are in-memory** (per-instance). For multi-replica deployments, swap the `TaskStore` implementation for Redis/BullMQ behind the same interface.
- **GPU** throughput is the win for the CUDA image; on CPU, ONNX Runtime already saturates cores, so concurrency mainly bounds memory.
