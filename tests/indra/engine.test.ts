import { beforeAll, describe, expect, test } from "bun:test";

import type { IndraPipeline } from "../../src/indra/pipeline.js";
import { item } from "../../src/indra/util.js";
import { CASES, loadCase, loadGray, makePipeline, tokenF1 } from "./golden.js";

let plain: IndraPipeline;
let withMath: IndraPipeline;

beforeAll(async () => {
  plain = await makePipeline(false, "canvas-native");
  withMath = await makePipeline(true, "canvas-native");
});

const OCR_CASES = CASES.filter((name) => item(loadCase(name).pages, 0).source === "ocr");

// canvas-native resizes and warps in plain float math, not OpenCV's fixed-point
// code, so it is held to token-F1 >= 0.98 against the reference (Q9), not exact text.
describe("canvas-native engine stays close to the reference", () => {
  test.each(OCR_CASES)("%s", async (name) => {
    const c = loadCase(name);
    const want = item(c.pages, 0);
    const pipeline = c.options["math"] ? withMath : plain;
    const page = await pipeline.page(await loadGray(item(c.gray, 0)));
    expect(page.rotation).toBe(want.rotation);
    expect(Math.abs(page.skew - want.skew)).toBeLessThanOrEqual(0.1);
    expect(tokenF1(page.text, want.text)).toBeGreaterThanOrEqual(0.98);
  });
});
