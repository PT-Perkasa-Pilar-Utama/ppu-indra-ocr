import { beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import * as ort from "onnxruntime-node";
import { join } from "path";
import { ImageProcessor, cv } from "ppu-ocv";

import { opencvImageOps } from "../../src/core/opencv-image-ops.js";
import type { IndraMeta } from "../../src/indra/pipeline.js";
import { IndraPipeline } from "../../src/indra/pipeline.js";
import { item } from "../../src/indra/util.js";
import { CASES, MODELS_DIR, event, events, field, loadCase, loadGray } from "./golden.js";

type GoldenLine = { text: string; conf: number; toks: [number | string, number][] };

let pipeline: IndraPipeline;

async function session(file: string): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(join(MODELS_DIR, file));
}

beforeAll(async () => {
  await ImageProcessor.initRuntime();
  // SAFETY: meta.json ships with the models in this shape.
  const meta = JSON.parse(readFileSync(join(MODELS_DIR, "meta.json"), "utf8")) as IndraMeta;
  pipeline = new IndraPipeline(
    ort.Tensor,
    {
      det: await session("det.onnx"),
      rec: await session("rec.onnx"),
      script: await session("script.onnx"),
      recCjk: await session("rec_cjk.onnx"),
    },
    meta,
    opencvImageOps(cv)
  );
});

const UPRIGHT = CASES.filter((name) => {
  const up = events(loadCase(name), "upright")[0];
  return up !== undefined && up["k"] === 0 && up["skew"] === 0;
});

describe("detect + recognize match the reference", () => {
  test.each(UPRIGHT)("%s", async (name) => {
    const c = loadCase(name);
    const gray = await loadGray(item(c.gray, 0));

    const boxes = await pipeline.detect(gray);
    const wantBoxes = field<number[][]>(event(c, "detect"), "boxes");
    expect(boxes.length).toBe(wantBoxes.length);
    for (const [i, b] of boxes.entries()) {
      for (let k = 0; k < 4; k++) expect(b[k]).toBeCloseTo(item(item(wantBoxes, i), k), 2);
    }

    // No input sha check here: the reference cv2 resizes through platform HALs
    // (Carotene and KleidiCV on arm64, IPP on x86), so its line crops differ from
    // OpenCV.js by a gray level or two. The bar is the recognized text.
    const lines = await pipeline.recognize(
      gray,
      boxes.map(([x0, y0, x1, y1]) => [x0, y0, x1, y1] as const)
    );
    const want = field<GoldenLine[]>(event(c, "recognize"), "lines");
    expect(lines.map((l) => l.text)).toEqual(want.map((l) => l.text));
    for (const [i, line] of lines.entries()) {
      const w = item(want, i);
      expect(Math.abs(line.conf - w.conf)).toBeLessThan(5e-3);
      expect(line.toks.map((t) => t[0])).toEqual(w.toks.map((t) => t[0]));
      for (const [k, t] of line.toks.entries()) expect(t[1]).toBeCloseTo(item(w.toks, k)[1], 2);
    }
  });
});
