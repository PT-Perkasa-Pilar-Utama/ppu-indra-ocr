import { afterAll, describe, expect, test } from "bun:test";

import { lineFeatures } from "../../src/indra/order-post.js";
import type { IndraPipeline } from "../../src/indra/pipeline.js";
import type { Box } from "../../src/indra/rec-preproc.js";
import { item } from "../../src/indra/util.js";
import type { ModelPreset } from "../../src/model-catalogue.js";
import {
  PRESET_CASES,
  event,
  field,
  loadCase,
  loadGray,
  makePipeline,
  releaseSessions,
} from "./golden.js";

const pipelines = new Map<string, Promise<IndraPipeline>>();

/** One pipeline per preset and math setting, made on first use. */
function pipelineFor(preset: ModelPreset, math: boolean): Promise<IndraPipeline> {
  const key = `${preset}/${math}`;
  let p = pipelines.get(key);
  if (!p) {
    p = makePipeline(math, "opencv", preset);
    pipelines.set(key, p);
  }
  return p;
}

// OCR-path cases: images, and PDFs that the reference OCR'd. Text-layer pages
// are checked in pdf.test.ts.
const OCR_CASES = PRESET_CASES.filter(
  ([preset, name]) => item(loadCase(name, preset).pages, 0).source === "ocr"
);

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
  test.each(OCR_CASES)("%s %s", (preset, name) => {
    const c = loadCase(name, preset);
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
  test.each(OCR_CASES)(
    "%s %s",
    async (preset, name) => {
      const c = loadCase(name, preset);
      const want = item(c.pages, 0);
      const pipeline = await pipelineFor(preset, Boolean(c.options["math"]));
      const page = await pipeline.page(await loadGray(item(c.gray, 0)));

      expect(page.rotation).toBe(want.rotation);
      expect(page.skew).toBe(want.skew);
      expect(page.vertical).toBe(want.vertical);
      expect(page.lines.map((l) => l.text)).toEqual(want.lines.map((l) => l.text));
      // A deskewed page is a uint8 warp, which the reference's cv2 runs through a
      // platform HAL: its pixels, and so a small box, can move by a pixel.
      const minIou = want.skew ? 0.9 : 0.98;
      for (const [i, line] of page.lines.entries()) {
        expect(iou(line.box, item(want.lines, i).box)).toBeGreaterThanOrEqual(minIou);
      }
      expect(page.blocks.map((b) => [b.label, b.text])).toEqual(
        want.blocks.map((b) => [b.label, b.text])
      );
      expect(page.markdown).toBe(want.markdown);
      expect(page.text).toBe(want.text);
      // PDF pages get a prefix from the document layer ("mode=ocr; ...").
      if (!c.input.endsWith(".pdf")) expect(page.note).toBe(want.note);
    },
    30_000
  );
});

afterAll(releaseSessions);
