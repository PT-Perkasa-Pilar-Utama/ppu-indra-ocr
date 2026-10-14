import { beforeAll, describe, expect, test } from "bun:test";

import { lineFeatures } from "../../src/indra/order-post.js";
import type { IndraPipeline } from "../../src/indra/pipeline.js";
import type { Box } from "../../src/indra/rec-preproc.js";
import { item } from "../../src/indra/util.js";
import { CASES, event, field, loadCase, loadGray, makePipeline } from "./golden.js";

let plain: IndraPipeline;
let withMath: IndraPipeline;

beforeAll(async () => {
  plain = await makePipeline();
  withMath = await makePipeline(true);
});

// OCR-path cases: images, and PDFs that the reference OCR'd. digital.auto
// joins with the text layer.
const OCR_CASES = CASES.filter((name) => item(loadCase(name).pages, 0).source === "ocr");

test("only digital.auto waits on a later port", () => {
  expect(CASES.filter((name) => !OCR_CASES.includes(name))).toEqual(["digital.auto"]);
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
    const pipeline = c.options["math"] ? withMath : plain;
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
