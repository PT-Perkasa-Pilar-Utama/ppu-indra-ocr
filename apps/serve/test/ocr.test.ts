import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(import.meta.dir, "..", "..", "..");
// Use the repo's models/ when it holds the real files (local), else the
// download cache that the root test preload fills (CI checks out LFS pointers).
const det = readFileSync(join(ROOT, "models", "det.onnx"))
  .subarray(0, 7)
  .toString();
if (det !== "version") process.env.MODEL_DIR = join(ROOT, "models");

const { IndraOcrService } = await import("ppu-indra-ocr");
const { app } = await import("../src/app.js");
const { shutdownService } = await import("../src/core/service.js");

const image = readFileSync(join(ROOT, "assets", "tilted.png"));
const pdf = readFileSync(join(ROOT, "tests", "fixtures", "pages", "digital.pdf"));
const dataUri = `data:image/png;base64,${image.toString("base64")}`;

const json = (body: unknown) =>
  ({
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  }) as const;

const upload = (
  bytes: Uint8Array,
  name: string,
  type: string,
  fields: Record<string, string> = {}
) => {
  const form = new FormData();
  form.append("file", new File([new Uint8Array(bytes)], name, { type }));
  for (const [k, v] of Object.entries(fields)) form.append(k, v);
  return { method: "POST", body: form } as const;
};

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// Only the download here: it fits the default hook timeout. Models load
// lazily on the first request; the inference tests carry longer timeouts.
beforeAll(async () => {
  if (!process.env.MODEL_DIR) await IndraOcrService.downloadModels();
});

afterAll(async () => {
  await shutdownService();
});

describe("POST /v1/ocr", () => {
  test("an image upload returns one page with markdown and metadata", async () => {
    const res = await app.request("/v1/ocr", upload(image, "tilted.png", "image/png"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("success");
    expect(body.data.pages).toHaveLength(1);
    expect(body.data.pages[0].text.length).toBeGreaterThan(0);
    expect(body.data.pages[0].skew).toBe(4.5);
    expect(body.metadata.pages).toBe(1);
    expect(body.metadata.speed).toBeGreaterThan(0);
    expect(body.metadata.engine).toBe("opencv");
  }, 30_000);

  test("a PDF upload reads the text layer", async () => {
    const res = await app.request(
      "/v1/ocr",
      upload(pdf, "digital.pdf", "application/pdf", { mode: "auto" })
    );
    expect(res.status).toBe(200);
    const page = (await res.json()).data.pages[0];
    expect(page.source).toBe("text");
    expect(page.markdown).toContain("## Springs of the Eastern Ridge");
  }, 30_000);

  test("a JSON data: URI source works; bad options are 400", async () => {
    const res = await app.request("/v1/ocr", json({ source: dataUri, engine: "opencv" }));
    expect(res.status).toBe(200);
    expect((await res.json()).data.pages).toHaveLength(1);
    expect((await app.request("/v1/ocr", json({ source: dataUri, mode: "fast" }))).status).toBe(
      400
    );
    expect(
      (await app.request("/v1/ocr", upload(pdf, "d.pdf", "application/pdf", { maxPages: "0" })))
        .status
    ).toBe(400);
  }, 30_000);
});

describe("POST /v1/detect", () => {
  test("returns line boxes with a count", async () => {
    const res = await app.request("/v1/detect", upload(image, "tilted.png", "image/png"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.data.boxes.length).toBeGreaterThan(0);
    expect(body.data.boxes[0].box).toHaveLength(4);
    expect(body.metadata.count).toBe(body.data.boxes.length);
  }, 30_000);
});

describe("POST /v1/ocr/batch", () => {
  test("returns one page per source; settle reports failures", async () => {
    const res = await app.request("/v1/ocr/batch", json({ sources: [dataUri, dataUri] }));
    expect(res.status).toBe(200);
    expect((await res.json()).data.results).toHaveLength(2);
    const settled = await app.request(
      "/v1/ocr/batch",
      json({
        sources: [dataUri, `data:application/pdf;base64,${pdf.toString("base64")}`],
        settle: true,
      })
    );
    expect((await settled.json()).data.results.map((r: { status: string }) => r.status)).toEqual([
      "fulfilled",
      "rejected",
    ]);
  }, 60_000);

  test("empty sources array is a validation error", async () => {
    const res = await app.request("/v1/ocr/batch", json({ sources: [] }));
    expect(res.status).toBe(400);
  });
});

describe("POST /v1/ocr/stream", () => {
  test("emits one SSE event per image then done", async () => {
    const res = await app.request("/v1/ocr/stream", json({ sources: [dataUri, dataUri] }));
    expect(res.status).toBe(200);
    const text = await res.text();
    expect((text.match(/event: fulfilled/g) ?? []).length).toBe(2);
    expect(text).toContain("event: done");
  }, 30_000);
});

describe("async tasks lifecycle", () => {
  test("submit, poll, result", async () => {
    const submit = await app.request("/v1/ocr/async", json({ sources: [dataUri] }));
    expect(submit.status).toBe(202);
    const { data } = await submit.json();
    let status = "queued";
    for (let i = 0; i < 50 && status !== "done"; i++) {
      status = (await (await app.request(`/v1/tasks/${data.taskId}`)).json()).data.status;
      if (status !== "done") await sleep(100);
    }
    expect(status).toBe("done");
    const result = await app.request(`/v1/tasks/${data.taskId}/result`);
    expect((await result.json()).data.results).toHaveLength(1);
  }, 30_000);

  test("unknown task id is 404; DELETE cancels a task", async () => {
    const id = crypto.randomUUID();
    expect((await app.request(`/v1/tasks/${id}`)).status).toBe(404);
    const submit = await app.request("/v1/ocr/async", json({ sources: [dataUri] }));
    const { data } = await submit.json();
    const del = await app.request(`/v1/tasks/${data.taskId}`, { method: "DELETE" });
    expect((await del.json()).data.status).toBe("cancelled");
  }, 30_000);
});
