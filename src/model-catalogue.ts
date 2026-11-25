// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/** The model files of Indra OCR, by role. */
export type ModelFile =
  | "det"
  | "rec"
  | "recCjk"
  | "recRtl"
  | "recWorld"
  | "script"
  | "orient"
  | "order"
  | "meta"
  | "lm"
  | "lmCjk"
  | "lmWorld"
  | "mathCls"
  | "mathEnc"
  | "mathDecInit"
  | "mathDecStep"
  | "mathDecStepKv"
  | "mathMeta"
  | "mathAccept"
  | "mathGate"
  | "mathSpanImg"
  | "mathSpanImgMeta"
  | "hw"
  | "recHw"
  | "tsrRel"
  | "tsrSplit"
  | "tsrRefine"
  | "tableVerify";

/** File name of each model in the Indra OCR model repository. */
export const MODEL_FILES: Readonly<Record<ModelFile, string>> = {
  det: "det.onnx",
  rec: "rec.onnx",
  recCjk: "rec_cjk.onnx",
  recRtl: "rec_rtl.onnx",
  recWorld: "rec_world.onnx",
  script: "script.onnx",
  orient: "orient.onnx",
  order: "order.onnx",
  meta: "meta.json",
  lm: "lm.npz",
  lmCjk: "lm_cjk.npz",
  lmWorld: "lm_world.npz",
  mathCls: "mathcls.onnx",
  mathEnc: "math_enc.onnx",
  mathDecInit: "math_dec_init.onnx",
  mathDecStep: "math_dec_step.onnx",
  mathDecStepKv: "math_dec_step_kv.onnx",
  mathMeta: "math_meta.json",
  mathAccept: "math_accept.json",
  mathGate: "math_gate.json",
  mathSpanImg: "math_span_img.onnx",
  mathSpanImgMeta: "math_span_img.json",
  hw: "hw.onnx",
  recHw: "rec_hw.onnx",
  tsrRel: "tsr_rel.onnx",
  tsrSplit: "tsr_split.onnx",
  tsrRefine: "tsr_refine.onnx",
  tableVerify: "table_verify.json",
};

/** The files every service loads when it starts. */
export const BASE_FILES: readonly ModelFile[] = ["det", "rec", "script", "orient", "order", "meta"];

/**
 * The recognizer of each script group, by the group name in meta.json
 * (`vocab_<group>`). One loads on the first line routed to its group.
 */
/** The recognizer file of each script group. */
export type GroupFiles = { cjk: "recCjk"; rtl: "recRtl"; world: "recWorld" };

export const GROUP_FILES: Readonly<GroupFiles> = {
  cjk: "recCjk",
  rtl: "recRtl",
  world: "recWorld",
};

/**
 * The n-gram language model of each recognizer ("base" is rec.onnx). One
 * loads on the first line its recognizer is unsure of, when `lm` is on.
 */
/** The language model file of each recognizer. */
export type LmFiles = { base: "lm"; cjk: "lmCjk"; world: "lmWorld" };

export const LM_FILES: Readonly<LmFiles> = {
  base: "lm",
  cjk: "lmCjk",
  world: "lmWorld",
};

/**
 * The math files: downloaded only when math is on. A preset has one decoder
 * step graph: math_dec_step_kv.onnx (per-layer caches, v1.2) or
 * math_dec_step.onnx (stacked caches). The accept head, the read gate and the
 * inline formula tagger are optional.
 */
export const MATH_FILES: readonly ModelFile[] = [
  "mathCls",
  "mathEnc",
  "mathDecInit",
  "mathDecStep",
  "mathDecStepKv",
  "mathMeta",
  "mathAccept",
  "mathGate",
  "mathSpanImg",
  "mathSpanImgMeta",
];

/** Table structure files (v1.2): tsr_split and tsr_refine only work with tsr_rel. */
export const TABLE_FILES: readonly ModelFile[] = ["tsrRel", "tsrSplit", "tsrRefine", "tableVerify"];

/** A model set by its Indra OCR release: the default and the legacy ones. */
export type ModelPreset = "v1.2" | "v1.1" | "v3.1";

/** Where a preset's files live and which files it has. */
export type PresetInfo = {
  /** Commit of datalab-to/indra-ocr, pinned so an upstream change cannot change results. */
  revision: string;
  /** The Hugging Face tag at that commit. */
  tag: string;
  /** Datalab's repository at the revision. */
  baseUrl: string;
  /** This repository's git LFS mirror of the same files (see MODEL_MIRROR_BASE_URL). */
  mirrorUrl: string;
  /** The files the preset has, math included. */
  files: readonly ModelFile[];
};

const HF = "https://huggingface.co/datalab-to/indra-ocr/resolve";

/** The math files of the v1.1 and v3.1 presets. */
const LEGACY_MATH_FILES: readonly ModelFile[] = [
  "mathCls",
  "mathEnc",
  "mathDecInit",
  "mathDecStep",
  "mathMeta",
];

/**
 * Mirror of the model files in this repository's models/ folder (git LFS),
 * used only when `PPU_INDRA_OCR_MODEL_MIRROR` is set.
 *
 * It must be the `github.com/<owner>/<repo>/raw/<ref>` form: that one
 * redirects to the LFS media host. `raw.githubusercontent.com` serves the LFS
 * pointer text instead. Off by default: LFS bandwidth has a monthly budget.
 */
export const MODEL_MIRROR_BASE_URL =
  "https://github.com/PT-Perkasa-Pilar-Utama/ppu-indra-ocr/raw/main/models";

/**
 * The model sets, by Indra OCR release:
 *
 * - `v1.2` (default): Indra OCR 1.2, Hugging Face tag v1.2.0. Retrained
 *   printed and handwriting recognizers (rec_hw.onnx for handwritten lines),
 *   table structure models, an order model that reads line thumbnails, and
 *   math gates. About 43 MB of networks (68 MB with math) plus 46 MB of
 *   language models. Its license sets the revenue and funding limits at
 *   USD 5,000,000.
 * - `v1.1`: Indra OCR 1.1, Hugging Face tag v1.1.0. Handwriting,
 *   80+ languages (recognizers for CJK, right-to-left and other scripts),
 *   pages at any angle, vertical writing, and language models for unsure
 *   lines. About 30 MB of networks, plus 46 MB of language models.
 * - `v3.1`: the first release (Indra OCR 1.0), Hugging Face tag v3.1. Latin,
 *   Cyrillic, Greek and Chinese. About 18 MB.
 */
export const MODEL_PRESETS: Readonly<Record<ModelPreset, PresetInfo>> = {
  "v1.2": {
    revision: "a04fa100e778192aa54d040a983983c6d3ee4ce5",
    tag: "v1.2.0",
    baseUrl: `${HF}/a04fa100e778192aa54d040a983983c6d3ee4ce5`,
    mirrorUrl: `${MODEL_MIRROR_BASE_URL}/v1.2`,
    files: [
      ...BASE_FILES,
      "recCjk",
      "recRtl",
      "recWorld",
      "hw",
      "recHw",
      "lm",
      "lmCjk",
      "lmWorld",
      ...TABLE_FILES,
      ...MATH_FILES.filter((f) => f !== "mathDecStep"),
    ],
  },
  "v1.1": {
    revision: "a1283c4b42a273c09480ad50e71461daff50fcc7",
    tag: "v1.1.0",
    baseUrl: `${HF}/a1283c4b42a273c09480ad50e71461daff50fcc7`,
    mirrorUrl: `${MODEL_MIRROR_BASE_URL}/v1.1`,
    files: [
      ...BASE_FILES,
      "recCjk",
      "recRtl",
      "recWorld",
      "lm",
      "lmCjk",
      "lmWorld",
      ...LEGACY_MATH_FILES,
    ],
  },
  "v3.1": {
    revision: "d7c839c892bfe4b8993373db64c62f154d05d70b",
    tag: "v3.1",
    baseUrl: `${HF}/d7c839c892bfe4b8993373db64c62f154d05d70b`,
    mirrorUrl: MODEL_MIRROR_BASE_URL,
    files: [...BASE_FILES, "recCjk", ...LEGACY_MATH_FILES],
  },
};

/** The preset a service uses when the options name none. */
export const DEFAULT_PRESET: ModelPreset = "v1.2";

/** The model revision of the default preset (commit of datalab-to/indra-ocr). */
export const MODEL_REVISION: string = MODEL_PRESETS[DEFAULT_PRESET].revision;

/**
 * Base URL for the default preset's files: Datalab's Hugging Face repository
 * at {@link MODEL_REVISION}. The weights are Datalab's, under their modified
 * OpenRAIL-M license, not this package's license.
 */
export const MODEL_BASE_URL: string = MODEL_PRESETS[DEFAULT_PRESET].baseUrl;

/** Whether a name is a model preset. */
export function isModelPreset(name: string): name is ModelPreset {
  return Object.hasOwn(MODEL_PRESETS, name);
}

/**
 * The files of a preset a service can load: without the math models unless
 * `math`, and without the language models when `lm` is false.
 */
export function presetFiles(
  preset: ModelPreset = DEFAULT_PRESET,
  { math = false, lm = true }: { math?: boolean; lm?: boolean } = {}
): ModelFile[] {
  const lms = new Set<ModelFile>(Object.values(LM_FILES));
  return MODEL_PRESETS[preset].files.filter(
    (role) => (math || !MATH_FILES.includes(role)) && (lm || !lms.has(role))
  );
}

/** Default URL of a model file in a preset. Throws for a file the preset does not have. */
export function defaultModelUrl(role: ModelFile, preset: ModelPreset = DEFAULT_PRESET): string {
  const info = MODEL_PRESETS[preset];
  if (!info.files.includes(role)) {
    throw new Error(`model preset ${preset} has no ${MODEL_FILES[role]}`);
  }
  return `${info.baseUrl}/${MODEL_FILES[role]}`;
}

/**
 * Rewrite a built-in model URL onto the preset's mirror folder. Returns null
 * when the mirror is off, or when the URL is not a preset's: a caller's own
 * model URL is never redirected. The environment is read on every call, so a
 * test can flip it.
 */
export function mirrorUrl(url: string): string | null {
  const enabled = globalThis.process?.env?.PPU_INDRA_OCR_MODEL_MIRROR;
  if (!enabled || enabled === "0" || enabled === "false") return null;
  for (const info of Object.values(MODEL_PRESETS)) {
    if (url.startsWith(`${info.baseUrl}/`)) return info.mirrorUrl + url.slice(info.baseUrl.length);
  }
  return null;
}
