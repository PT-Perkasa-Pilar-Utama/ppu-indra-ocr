import { afterAll, describe, expect, test } from "bun:test";

import type { IndraPipeline } from "../../src/indra/pipeline.js";
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
} from "./golden.js";

type GoldenLine = { text: string; conf: number; toks: [number | string, number][] };

const pipelines = new Map<string, Promise<IndraPipeline>>();
function pipelineFor(preset: ModelPreset, math: boolean): Promise<IndraPipeline> {
  const key = `${preset}/${math}`;
  let p = pipelines.get(key);
  if (!p) {
    p = makePipeline(math, "opencv", preset);
    pipelines.set(key, p);
  }
  return p;
}

// Pages the OCR read as they are: no turn and no deskew.
const UPRIGHT = PRESET_CASES.filter(([preset, name]) => {
  const turn = events(loadCase(name, preset), "turn")[0];
  return turn !== undefined && turn["k"] === 0 && turn["skew"] === 0;
});

describe("detect + recognize match the reference", () => {
  test.each(UPRIGHT)("%s %s", async (preset, name) => {
    const c = loadCase(name, preset);
    const pipeline = await pipelineFor(preset, Boolean(c.options["math"]));
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

afterAll(releaseSessions);
