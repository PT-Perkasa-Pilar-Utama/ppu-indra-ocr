import { beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import * as ort from "onnxruntime-node";
import { join } from "path";
import { ImageProcessor, cv } from "ppu-ocv";

import { opencvImageOps } from "../../src/core/opencv-image-ops.js";
import type { IndraMeta } from "../../src/indra/types.js";
import { IndraPipeline } from "../../src/indra/pipeline.js";
import { item } from "../../src/indra/util.js";
import { CASES, MODELS_DIR, event, events, field, loadCase, loadGray, sha256 } from "./golden.js";

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
      orient: await session("orient.onnx"),
    },
    meta,
    opencvImageOps(cv)
  );
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
