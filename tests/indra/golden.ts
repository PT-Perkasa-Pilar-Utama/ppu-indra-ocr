import { loadImage, createCanvas } from "@napi-rs/canvas";
import { createHash } from "crypto";
import { readFileSync } from "fs";
import { join } from "path";

import * as ort from "onnxruntime-node";
import { ImageProcessor, cv } from "ppu-ocv";

import { opencvImageOps } from "../../src/core/opencv-image-ops.js";
import type { Gray } from "../../src/indra/image.js";
import { NgramLM } from "../../src/indra/lm.js";
import type { MathMeta } from "../../src/indra/math.js";
import { MathReader } from "../../src/indra/math.js";
import { IndraPipeline } from "../../src/indra/pipeline.js";
import { tsImageOps } from "../../src/indra/ts-image-ops.js";
import type { IndraMeta } from "../../src/indra/types.js";
import type { ModelSources } from "../../src/interface.js";
import type { ModelFile, ModelPreset } from "../../src/model-catalogue.js";
import {
  GROUP_FILES,
  LM_FILES,
  MODEL_FILES,
  MODEL_PRESETS,
  defaultModelUrl,
} from "../../src/model-catalogue.js";
import { fetchAndCacheResource } from "../../src/processor/model-cache.js";

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
  vertical: boolean;
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

const LEGACY_CASES = [
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

/** The golden cases of each preset: v1.1 (the default) adds the 1.1 features. */
export const CASES_BY_PRESET: Readonly<Record<ModelPreset, readonly string[]>> = {
  "v1.1": [
    ...LEGACY_CASES,
    "multiscript",
    "hebrew",
    "vertical",
    "tilted-30",
    "sideways.auto",
    "rotate-90.auto",
    "rtl-text.auto",
    "vertical-text.auto",
  ],
  "v3.1": LEGACY_CASES,
};

/** The default preset's cases. */
export const CASES: readonly string[] = CASES_BY_PRESET["v1.1"];

/** Every (preset, case) pair. */
export const PRESET_CASES: [ModelPreset, string][] = (["v1.1", "v3.1"] as const).flatMap((p) =>
  CASES_BY_PRESET[p].map((c): [ModelPreset, string] => [p, c])
);

export function loadCase(name: string, preset: ModelPreset = "v1.1"): GoldenCase {
  const dir = preset === "v1.1" ? GOLDEN_DIR : join(GOLDEN_DIR, preset);
  // SAFETY: the file is written by scripts/golden/gen.py in this shape.
  return JSON.parse(readFileSync(join(dir, `${name}.json`), "utf8")) as GoldenCase;
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

const LFS_POINTER = "version https://git-lfs.github.com/spec/v1";

/** The mirror folder of a preset's files. */
export function modelsDir(preset: ModelPreset = "v1.1"): string {
  return preset === "v1.1" ? join(MODELS_DIR, "v1.1") : MODELS_DIR;
}

/**
 * Model sources for tests: the mirrored file in models/ when it is real, else
 * the default URL (served from the disk cache that tests/preload-warm-cache.ts
 * fills). CI checks out LFS pointers, not the binaries.
 */
export function testModelSources(preset: ModelPreset = "v1.1"): ModelSources {
  const out: ModelSources = {};
  for (const role of MODEL_PRESETS[preset].files) {
    const file = join(modelsDir(preset), MODEL_FILES[role]);
    if (!readFileSync(file).subarray(0, LFS_POINTER.length).toString().startsWith(LFS_POINTER))
      out[role] = file;
  }
  return out;
}

export async function modelBytes(
  role: ModelFile,
  preset: ModelPreset = "v1.1"
): Promise<Uint8Array> {
  const local = testModelSources(preset)[role];
  if (typeof local === "string") return readFileSync(local);
  return new Uint8Array(await fetchAndCacheResource(defaultModelUrl(role, preset)));
}

const opened: ort.InferenceSession[] = [];

/** A session on one model file, with the CPU arena off as the service sets it. Freed by releaseSessions(). */
export async function session(
  role: ModelFile,
  preset: ModelPreset = "v1.1"
): Promise<ort.InferenceSession> {
  const s = await ort.InferenceSession.create(await modelBytes(role, preset), {
    enableCpuMemArena: false,
  });
  opened.push(s);
  return s;
}

/** Release every session this test file opened. Call it in afterAll. */
export async function releaseSessions(): Promise<void> {
  await Promise.all(opened.splice(0).map((s) => s.release()));
}

async function readJson<T>(role: ModelFile, preset: ModelPreset): Promise<T> {
  // SAFETY: the JSON files ship with the models; the caller names their shape.
  return JSON.parse(new TextDecoder().decode(await modelBytes(role, preset))) as T;
}

/**
 * The pipeline on a preset's mirrored models; `math` adds the math reader,
 * `engine` picks the image ops. Group recognizers and language models load
 * on first use, as in the service.
 */
export async function makePipeline(
  math = false,
  engine: "opencv" | "canvas-native" = "opencv",
  preset: ModelPreset = "v1.1"
): Promise<IndraPipeline> {
  await ImageProcessor.initRuntime();
  const ops = engine === "opencv" ? opencvImageOps(cv) : tsImageOps;
  const reader = math
    ? new MathReader(
        ort.Tensor,
        {
          cls: await session("mathCls", preset),
          enc: await session("mathEnc", preset),
          init: await session("mathDecInit", preset),
          step: await session("mathDecStep", preset),
        },
        await readJson<MathMeta>("mathMeta", preset),
        ops
      )
    : undefined;
  const files = MODEL_PRESETS[preset].files;
  const groups = Object.fromEntries(
    Object.entries(GROUP_FILES).flatMap(([name, role]) =>
      files.includes(role) ? [[name, () => session(role, preset)]] : []
    )
  );
  const lms = Object.fromEntries(
    Object.entries(LM_FILES).flatMap(([name, role]) =>
      files.includes(role)
        ? [[name, async () => NgramLM.fromNpz(await modelBytes(role, preset))]]
        : []
    )
  );
  return new IndraPipeline(
    ort.Tensor,
    {
      det: await session("det", preset),
      rec: await session("rec", preset),
      order: await session("order", preset),
      orient: await session("orient", preset),
      script: await session("script", preset),
      groups,
      lms,
    },
    await readJson<IndraMeta>("meta", preset),
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
