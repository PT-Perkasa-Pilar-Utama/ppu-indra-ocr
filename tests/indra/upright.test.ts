import { afterAll, describe, expect, test } from "bun:test";

import type { IndraPipeline } from "../../src/indra/pipeline.js";
import { DESKEW_MIN, estimateSkew, preview, rotatePage } from "../../src/indra/upright.js";
import { item } from "../../src/indra/util.js";
import type { ModelPreset } from "../../src/model-catalogue.js";
import {
  PRESET_CASES,
  event,
  events,
  field,
  loadCase,
  loadGray,
  makePipeline,
  releaseSessions,
  sha256,
} from "./golden.js";

const pipelines = new Map<ModelPreset, Promise<IndraPipeline>>();
function pipelineFor(preset: ModelPreset): Promise<IndraPipeline> {
  let p = pipelines.get(preset);
  if (!p) {
    p = makePipeline(false, "opencv", preset);
    pipelines.set(preset, p);
  }
  return p;
}

const WITH_ORIENT = PRESET_CASES.filter(
  ([preset, name]) => events(loadCase(name, preset), "orient.run").length > 0
);

describe("preview, skew, orient and straighten match the reference", () => {
  test.each(WITH_ORIENT)(
    "%s %s",
    async (preset, name) => {
      const c = loadCase(name, preset);
      const pipeline = await pipelineFor(preset);
      const ops = pipeline.imageOps;
      const gray = await loadGray(item(c.gray, 0));

      const small = preview(gray, ops);
      const pv = event(c, "preview");
      expect([small.height, small.width]).toEqual(field<number[]>(pv, "shape"));
      // Downscales with INTER_AREA take no platform HAL path: byte-exact.
      expect(sha256(small.data)).toBe(field<string>(pv, "sha"));

      const a = estimateSkew(small, ops);
      expect(a).toBeCloseTo(field<number>(event(c, "estimate_skew"), "angle"), 6);

      const { turns, probs } = await pipeline.orient(
        Math.abs(a) >= DESKEW_MIN ? rotatePage(small, a, ops) : small
      );
      const want = field<number[]>(event(c, "orient.run"), "p");
      for (const [i, p] of probs.entries()) expect(p).toBeCloseTo(item(want, i), 3);

      const turn = event(c, "turn");
      expect(turns).toBe(field<number>(turn, "k"));
      const { image, skew } = pipeline.straighten(gray, turns, a, small);
      expect(skew).toBeCloseTo(field<number>(turn, "skew"), 6);
      expect([image.height, image.width]).toEqual(field<number[]>(turn, "shape"));
      // A deskew warps uint8 pixels, which the reference cv2 does through a
      // platform HAL; only unskewed pages (no warp, or a pure quarter turn) are byte-exact.
      if (skew === 0) expect(sha256(image.data)).toBe(field<string>(turn, "sha"));
    },
    30_000
  );
});

afterAll(releaseSessions);
