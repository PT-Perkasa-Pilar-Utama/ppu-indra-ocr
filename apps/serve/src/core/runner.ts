import type { BatchItemResult, DetBox, Page, ProcessingEngine } from "ppu-indra-ocr";
import { convertPdf, isPdf } from "ppu-indra-ocr/pdf";
import { config } from "./config.js";
import { badRequest } from "./errors.js";
import { resolveSource } from "./input.js";
import type { BatchOcrBody, OcrOptions } from "./schemas.js";
import { getService, queue } from "./service.js";

export type OcrMeta = { engine: ProcessingEngine; ms: number };
export type OcrResponse = { result: { pages: Page[] }; meta: OcrMeta };

const round1 = (n: number): number => Math.round(n * 10) / 10;

/** OCR one image or PDF through the bounded inference queue. A PDF gives one page each. */
export async function runOcr(input: ArrayBuffer, opts: OcrOptions): Promise<OcrResponse> {
  const engine = opts.engine ?? config.defaultEngine;
  const svc = await getService(engine);
  const start = performance.now();
  const pages = isPdf(input)
    ? await queue.run(() =>
        convertPdf(svc, input, {
          mode: opts.mode,
          dpi: config.pdfDpi,
          maxPages: Math.min(opts.maxPages ?? config.maxPdfPages, config.maxPdfPages),
        })
      )
    : [await queue.run(() => svc.convert(input))];
  return { result: { pages }, meta: { engine, ms: round1(performance.now() - start) } };
}

/** One detected line: its box (x0, y0, x1, y1) and score. */
export type DetectedLine = { box: number[]; score: number };

export type DetectResponse = { result: { boxes: DetectedLine[] }; meta: OcrMeta };

const toDetected = ([x0, y0, x1, y1, score]: DetBox): DetectedLine => ({
  box: [x0, y0, x1, y1],
  score,
});

/** Detect text lines only (no recognition) through the bounded inference queue. */
export async function runDetect(image: ArrayBuffer, opts: OcrOptions): Promise<DetectResponse> {
  const engine = opts.engine ?? config.defaultEngine;
  const svc = await getService(engine);
  const start = performance.now();
  const boxes = await queue.run(() => svc.detect(image));
  return {
    result: { boxes: boxes.map(toDetected) },
    meta: { engine, ms: round1(performance.now() - start) },
  };
}

/** Resolve a batch of `sources` to image buffers, bounded by MAX_BATCH_IMAGES. */
export async function resolveBatch(sources: string[]): Promise<ArrayBuffer[]> {
  if (sources.length > config.maxBatchImages) {
    throw badRequest(`Too many sources (max ${config.maxBatchImages})`);
  }
  return Promise.all(sources.map(resolveSource));
}

export type BatchResponse = {
  results: Page[] | BatchItemResult<Page>[];
  meta: { engine: ProcessingEngine };
};

/** Convert a batch of images; convertBatch applies its own bounded concurrency. */
export async function runBatch(
  images: ArrayBuffer[],
  body: BatchOcrBody,
  signal?: AbortSignal
): Promise<BatchResponse> {
  const engine = body.engine ?? config.defaultEngine;
  const svc = await getService(engine);
  const concurrency = body.concurrency ?? config.concurrency;
  const results = body.settle
    ? await svc.convertBatch(images, { concurrency, settle: true, signal })
    : await svc.convertBatch(images, { concurrency, signal });
  return { results, meta: { engine } };
}

/** Stream batch results as each image finishes (completion order). */
export async function* streamBatch(
  images: ArrayBuffer[],
  body: BatchOcrBody
): AsyncGenerator<BatchItemResult<Page>> {
  const svc = await getService(body.engine ?? config.defaultEngine);
  yield* svc.convertStream(images, {
    concurrency: body.concurrency ?? config.concurrency,
    settle: body.settle ?? false,
  });
}
