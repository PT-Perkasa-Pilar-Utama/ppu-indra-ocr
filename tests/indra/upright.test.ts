import { afterAll, beforeAll, describe, expect, test } from "bun:test";

import type { IndraPipeline } from "../../src/indra/pipeline.js";
import { item } from "../../src/indra/util.js";
import {
  CASES,
  event,
  events,
  field,
  loadCase,
  loadGray,
  makePipeline,
  releaseSessions,
  sha256,
} from "./golden.js";

let pipeline: IndraPipeline;

beforeAll(async () => {
  pipeline = await makePipeline();
});

const WITH_ORIENT = CASES.filter((name) => events(loadCase(name), "orient.run").length > 0);

describe("orient + straighten match the reference", () => {
  test.each(WITH_ORIENT)("%s", async (name) => {
    const c = loadCase(name);
    const gray = await loadGray(item(c.gray, 0));
    const { turns, probs } = await pipeline.orient(gray);
    const want = field<number[]>(event(c, "orient.run"), "p");
    for (const [i, p] of probs.entries()) expect(p).toBeCloseTo(item(want, i), 4);

    const up = event(c, "upright");
    expect(turns).toBe(field<number>(up, "k"));
    const { image, skew } = pipeline.straighten(gray, turns);
    expect(skew).toBeCloseTo(field<number>(up, "skew"), 6);
    expect([image.height, image.width]).toEqual(field<number[]>(up, "shape"));
    // A deskew warps uint8 pixels, which the reference cv2 does through a
    // platform HAL; only unskewed pages (no warp, or a pure quarter turn) are byte-exact.
    if (skew === 0) expect(sha256(image.data)).toBe(field<string>(up, "sha"));
  });
});

afterAll(releaseSessions);
