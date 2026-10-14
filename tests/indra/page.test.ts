import { beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import * as ort from "onnxruntime-node";
import { join } from "path";
import { ImageProcessor, cv } from "ppu-ocv";

import { opencvImageOps } from "../../src/core/opencv-image-ops.js";
import { lineFeatures } from "../../src/indra/order-post.js";
import { IndraPipeline } from "../../src/indra/pipeline.js";
import type { Box } from "../../src/indra/rec-preproc.js";
import type { IndraMeta } from "../../src/indra/types.js";
import { item } from "../../src/indra/util.js";
import { CASES, MODELS_DIR, event, field, loadCase, loadGray } from "./golden.js";

let pipeline: IndraPipeline;

beforeAll(async () => {
  await ImageProcessor.initRuntime();
  // SAFETY: meta.json ships with the models in this shape.
  const meta = JSON.parse(readFileSync(join(MODELS_DIR, "meta.json"), "utf8")) as IndraMeta;
  const session = (file: string): Promise<ort.InferenceSession> =>
    ort.InferenceSession.create(join(MODELS_DIR, file));
  pipeline = new IndraPipeline(
    ort.Tensor,
    {
      det: await session("det.onnx"),
      rec: await session("rec.onnx"),
      order: await session("order.onnx"),
      orient: await session("orient.onnx"),
      script: await session("script.onnx"),
      recCjk: await session("rec_cjk.onnx"),
    },
    meta,
    opencvImageOps(cv)
  );
});

// OCR-path cases: images, and PDFs that the reference OCR'd. math.on joins
// with the math reader; digital.auto with the text layer.
const OCR_CASES = CASES.filter((name) => {
  const c = loadCase(name);
  return !c.options["math"] && item(c.pages, 0).source === "ocr";
});

test("only math.on and digital.auto wait on later ports", () => {
  expect(CASES.filter((name) => !OCR_CASES.includes(name))).toEqual(["math.on", "digital.auto"]);
});

function iou(a: readonly number[], b: readonly number[]): number {
  const [ax0 = 0, ay0 = 0, ax1 = 0, ay1 = 0] = a;
  const [bx0 = 0, by0 = 0, bx1 = 0, by1 = 0] = b;
  const iw = Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0));
  const ih = Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0));
  const inter = iw * ih;
  const union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter;
  return union > 0 ? inter / union : 1;
}

describe("line features match the reference", () => {
  test.each(OCR_CASES)("%s", (name) => {
    const c = loadCase(name);
    const ro = event(c, "reading_order");
    const [w = 0, h = 0] = field<number[]>(ro, "wh");
    const boxes = field<number[][]>(ro, "boxes").map(([x0 = 0, y0 = 0, x1 = 0, y1 = 0]): Box => [
      x0,
      y0,
      x1,
      y1,
    ]);
    const feats = lineFeatures(boxes, w, h, field<string[]>(ro, "texts"));
    const want = field<number[][]>(event(c, "order.run"), "feats").flat();
    expect(feats.length).toBe(want.length);
    // Golden boxes are rounded to 3 decimals, so features match to about 1e-4.
    for (const [i, v] of feats.entries()) expect(Math.abs(v - item(want, i))).toBeLessThan(2e-4);
  });
});

describe("full page matches the reference", () => {
  test.each(OCR_CASES)("%s", async (name) => {
    const c = loadCase(name);
    const want = item(c.pages, 0);
    const page = await pipeline.page(await loadGray(item(c.gray, 0)));

    expect(page.rotation).toBe(want.rotation);
    expect(page.skew).toBe(want.skew);
    expect(page.lines.map((l) => l.text)).toEqual(want.lines.map((l) => l.text));
    for (const [i, line] of page.lines.entries()) {
      expect(iou(line.box, item(want.lines, i).box)).toBeGreaterThanOrEqual(0.98);
    }
    expect(page.blocks.map((b) => [b.label, b.text])).toEqual(
      want.blocks.map((b) => [b.label, b.text])
    );
    expect(page.markdown).toBe(want.markdown);
    expect(page.text).toBe(want.text);
    // PDF pages get a prefix from the document layer ("mode=ocr; ...").
    if (!c.input.endsWith(".pdf")) expect(page.note).toBe(want.note);
  });
});
