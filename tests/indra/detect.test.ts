import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import * as ort from "onnxruntime-node";
import { ImageProcessor, cv } from "ppu-ocv";

import { opencvImageOps } from "../../src/core/opencv-image-ops.js";
import { decode, pageToInput } from "../../src/indra/det-post.js";
import type { ImageOps } from "../../src/indra/image.js";
import { floats, item, output } from "../../src/indra/util.js";
import type { ModelPreset } from "../../src/model-catalogue.js";
import {
  PRESET_CASES,
  event,
  events,
  field,
  loadCase,
  loadGray,
  releaseSessions,
  session,
  sha256,
} from "./golden.js";

let ops: ImageOps;
const det = new Map<ModelPreset, ort.InferenceSession>();

beforeAll(async () => {
  await ImageProcessor.initRuntime();
  ops = opencvImageOps(cv);
  det.set("v1.1", await session("det", "v1.1"));
  det.set("v3.1", await session("det", "v3.1"));
});

// Cases whose page reaches detection unrotated and unskewed, so the golden
// gray page is the detector's page.
const UPRIGHT = PRESET_CASES.filter(([preset, name]) => {
  const turn = events(loadCase(name, preset), "turn")[0];
  return turn !== undefined && turn["k"] === 0 && turn["skew"] === 0;
});

describe("detection matches the reference", () => {
  test.each(UPRIGHT)("%s %s", async (preset, name) => {
    const c = loadCase(name, preset);
    const session = det.get(preset);
    if (!session) throw new Error(`no detector for ${preset}`);
    const gray = await loadGray(item(c.gray, 0));
    const passes = events(c, "detect_at");
    expect(events(c, "det.run").length).toBe(passes.length);
    for (const [i, pass] of passes.entries()) {
      const { input, scale } = pageToInput(gray, field<number>(pass, "long_side"), ops);
      // Detector input is byte-exact: INTER_AREA downscale takes no platform HAL path.
      expect(sha256(input.data)).toBe(field<string>(event(c, "det.run", i), "input_sha"));
      expect(scale).toBeCloseTo(field<number>(pass, "scale"), 7);

      const x = new ort.Tensor("float32", input.data, [1, 1, input.height, input.width]);
      const out = output(await session.run({ x }), "out");
      const [, , h = 0, w = 0] = out.dims;
      const boxes = decode(floats(out), h, w);
      const want = field<number[][]>(pass, "boxes");
      expect(boxes.length).toBe(want.length);
      for (const [j, box] of boxes.entries()) {
        const [x0, y0, x1, y1, score] = item(want, j);
        expect(box[0]).toBeCloseTo(x0 ?? NaN, 2);
        expect(box[1]).toBeCloseTo(y0 ?? NaN, 2);
        expect(box[2]).toBeCloseTo(x1 ?? NaN, 2);
        expect(box[3]).toBeCloseTo(y1 ?? NaN, 2);
        expect(box[4]).toBeCloseTo(score ?? NaN, 3);
      }
    }
  });
});

afterAll(releaseSessions);
