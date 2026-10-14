import { loadImage, createCanvas } from "@napi-rs/canvas";
import { createHash } from "crypto";
import { readFileSync } from "fs";
import { join } from "path";

import * as ort from "onnxruntime-node";
import { ImageProcessor, cv } from "ppu-ocv";

import { opencvImageOps } from "../../src/core/opencv-image-ops.js";
import type { Gray } from "../../src/indra/image.js";
import type { MathMeta } from "../../src/indra/math.js";
import { MathReader } from "../../src/indra/math.js";
import { IndraPipeline } from "../../src/indra/pipeline.js";
import { tsImageOps } from "../../src/indra/ts-image-ops.js";
import type { IndraMeta } from "../../src/indra/types.js";

/**
 * Goldens from the Python reference (scripts/golden/gen.py). Each case holds
 * the stage events in call order and the final page.
 */

export const GOLDEN_DIR: string = join(import.meta.dir, "..", "fixtures", "golden");
export const MODELS_DIR: string = join(import.meta.dir, "..", "..", "models");

export type GoldenEvent = { stage: string } & Record<string, unknown>;

export type GoldenPage = {
  source: string;
  note: string;
  rotation: number;
  skew: number;
  lines: { box: number[]; text: string; conf: number }[];
  blocks: { label: string; bbox: number[]; text: string }[];
  markdown: string | null;
  text: string;
};

export type GoldenCase = {
  case: string;
  input: string;
  gray: string[];
  options: Record<string, unknown>;
  events: GoldenEvent[];
  pages: GoldenPage[];
};

export const CASES = [
  "receipt",
  "tilted",
  "two-column",
  "rotated-90",
  "rotated-180",
  "table",
  "chinese",
  "math",
  "math.on",
  "digital.auto",
  "digital.ocr",
  "scanned.auto",
] as const;

export function loadCase(name: string): GoldenCase {
  // SAFETY: the file is written by scripts/golden/gen.py in this shape.
  return JSON.parse(readFileSync(join(GOLDEN_DIR, `${name}.json`), "utf8")) as GoldenCase;
}

export function events(c: GoldenCase, stage: string): GoldenEvent[] {
  return c.events.filter((e) => e.stage === stage);
}

/** The n-th event of a stage; throws when the reference never ran it. */
export function event(c: GoldenCase, stage: string, n = 0): GoldenEvent {
  const e = events(c, stage)[n];
  if (!e) throw new Error(`${c.case}: no ${stage} event #${n}`);
  return e;
}

/** A field of a golden event, typed by the caller. */
export function field<T>(e: GoldenEvent, key: string): T {
  if (!(key in e)) throw new Error(`${e.stage} event has no "${key}"`);
  // SAFETY: gen.py writes each stage's fields with fixed shapes; the caller names the shape.
  return e[key] as T;
}

/** The grayscale page the reference read, decoded from its lossless PNG. */
export async function loadGray(file: string): Promise<Gray> {
  const img = await loadImage(readFileSync(join(GOLDEN_DIR, file)));
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const rgba = ctx.getImageData(0, 0, img.width, img.height).data;
  const data = new Uint8Array(img.width * img.height);
  for (let i = 0; i < data.length; i++) data[i] = rgba[i * 4] ?? 0;
  return { data, width: img.width, height: img.height };
}

export function sha256(data: Float32Array | Uint8Array): string {
  return createHash("sha256")
    .update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))
    .digest("hex");
}

function session(file: string): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(join(MODELS_DIR, file));
}

function readJson<T>(file: string): T {
  // SAFETY: the JSON files ship with the models; the caller names their shape.
  return JSON.parse(readFileSync(join(MODELS_DIR, file), "utf8")) as T;
}

/** The pipeline on the mirrored models; `math` adds the math reader, `engine` picks the image ops. */
export async function makePipeline(
  math = false,
  engine: "opencv" | "canvas-native" = "opencv"
): Promise<IndraPipeline> {
  await ImageProcessor.initRuntime();
  const ops = engine === "opencv" ? opencvImageOps(cv) : tsImageOps;
  const reader = math
    ? new MathReader(
        ort.Tensor,
        {
          cls: await session("mathcls.onnx"),
          enc: await session("math_enc.onnx"),
          init: await session("math_dec_init.onnx"),
          step: await session("math_dec_step.onnx"),
        },
        readJson<MathMeta>("math_meta.json"),
        ops
      )
    : undefined;
  return new IndraPipeline(
    ort.Tensor,
    {
      det: await session("det.onnx"),
      rec: await session("rec.onnx"),
      order: await session("order.onnx"),
      orient: await session("orient.onnx"),
      script: await session("script.onnx"),
      recCjk: await session("rec_cjk.onnx"),
    },
    readJson<IndraMeta>("meta.json"),
    ops,
    { math: reader }
  );
}

/** Token-level F1 between two texts (whitespace tokens, order ignored). */
export function tokenF1(got: string, want: string): number {
  const count = (s: string): Map<string, number> => {
    const m = new Map<string, number>();
    for (const t of s.split(/\s+/).filter(Boolean)) m.set(t, (m.get(t) ?? 0) + 1);
    return m;
  };
  const g = count(got);
  const w = count(want);
  let hit = 0;
  for (const [t, n] of g) hit += Math.min(n, w.get(t) ?? 0);
  const ng = [...g.values()].reduce((a, b) => a + b, 0);
  const nw = [...w.values()].reduce((a, b) => a + b, 0);
  return ng + nw === 0 ? 1 : (2 * hit) / (ng + nw);
}
