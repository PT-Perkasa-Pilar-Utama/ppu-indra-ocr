import { afterAll, describe, expect, test } from "bun:test";
import { ImageProcessor, cv } from "ppu-ocv";

import { opencvImageOps } from "../../src/core/opencv-image-ops.js";
import { rotationMatrix } from "../../src/indra/image.js";
import type { Gray, ImageOps } from "../../src/indra/image.js";
import type { IndraPipeline } from "../../src/indra/pipeline.js";
import { tsImageOps } from "../../src/indra/ts-image-ops.js";
import {
  DESKEW_MIN,
  IDENTITY_TURN,
  estimateSkew,
  preview,
  rotatePage,
} from "../../src/indra/upright.js";
import { item } from "../../src/indra/util.js";
import type { ModelPreset } from "../../src/model-catalogue.js";
import {
  PRESET_CASES,
  event,
  events,
  field,
  finalRead,
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

      // The page's own turn: below ORIENT_MIN, or for a half turn, lines are
      // read each way first (pageTurns), and those reads log their own turns.
      const turn = event(finalRead(c), "turn");
      expect(turns === field<number>(turn, "k") || events(c, "read_half_turn").length > 0).toBe(
        true
      );
      expect(await pipeline.pageTurns(gray, a, small)).toBe(field<number>(turn, "k"));
      const { image, skew } = pipeline.straighten(gray, field<number>(turn, "k"), a, small);
      expect(skew).toBeCloseTo(field<number>(turn, "skew"), 6);
      expect([image.height, image.width]).toEqual(field<number[]>(turn, "shape"));
      // A deskew warps uint8 pixels, which the reference cv2 does through a
      // platform HAL; only unskewed pages (no warp, or a pure quarter turn) are byte-exact.
      if (skew === 0) expect(sha256(image.data)).toBe(field<string>(turn, "sha"));
    },
    30_000
  );
});

// The skew search skips the warp below IDENTITY_TURN degrees, which is
// correct only while warpZero returns the input bytes there.
describe("warpZero below IDENTITY_TURN returns its input", () => {
  // Odd sizes put the rotation center between pixels, and noise makes any
  // half-pixel shift change bytes.
  let seed = 1;
  const noise: Gray = {
    width: 801,
    height: 613,
    data: Uint8Array.from({ length: 801 * 613 }, () => {
      seed = (seed * 48271) % 2147483647;
      return seed & 255;
    }),
  };
  // -1.1102230246251565e-16 is what arange(-0.5, 0.5 + 1e-6, 0.1) scores for 0.
  const turns = [0, -1.1102230246251565e-16, IDENTITY_TURN, -IDENTITY_TURN];
  const backends: [string, () => Promise<ImageOps>][] = [
    ["canvas-native", async () => tsImageOps],
    [
      "opencv",
      async () => {
        await ImageProcessor.initRuntime();
        return opencvImageOps(cv);
      },
    ],
  ];

  test.each(backends)("%s", async (_, makeOps) => {
    const ops = await makeOps();
    const { width: w, height: h } = noise;
    for (const a of turns) {
      const out = ops.warpZero(noise, rotationMatrix(w / 2, h / 2, a), w, h);
      expect(out.data).toEqual(noise.data);
    }
  });
});

afterAll(releaseSessions);
