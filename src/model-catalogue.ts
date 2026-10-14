// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/** The model files of Indra OCR, by role. */
export type ModelFile =
  | "det"
  | "rec"
  | "recCjk"
  | "script"
  | "orient"
  | "order"
  | "meta"
  | "mathCls"
  | "mathEnc"
  | "mathDecInit"
  | "mathDecStep"
  | "mathMeta";

/** File name of each model in the Indra OCR model repository. */
export const MODEL_FILES: Readonly<Record<ModelFile, string>> = {
  det: "det.onnx",
  rec: "rec.onnx",
  recCjk: "rec_cjk.onnx",
  script: "script.onnx",
  orient: "orient.onnx",
  order: "order.onnx",
  meta: "meta.json",
  mathCls: "mathcls.onnx",
  mathEnc: "math_enc.onnx",
  mathDecInit: "math_dec_init.onnx",
  mathDecStep: "math_dec_step.onnx",
  mathMeta: "math_meta.json",
};

/** The files every service loads. */
export const BASE_FILES: readonly ModelFile[] = [
  "det",
  "rec",
  "recCjk",
  "script",
  "orient",
  "order",
  "meta",
];

/** The math files: downloaded only when math is on. */
export const MATH_FILES: readonly ModelFile[] = [
  "mathCls",
  "mathEnc",
  "mathDecInit",
  "mathDecStep",
  "mathMeta",
];

/**
 * The model revision this package is built and tested against: commit
 * d7c839c of datalab-to/indra-ocr (tag v3.1). Pinned to the commit, so an
 * upstream change (meta.json holds the vocabulary) cannot change results.
 */
export const MODEL_REVISION = "d7c839c892bfe4b8993373db64c62f154d05d70b";

/**
 * Base URL for model files: Datalab's Hugging Face repository at
 * {@link MODEL_REVISION}. The weights are Datalab's, under their modified
 * OpenRAIL-M license, not this package's license.
 */
export const MODEL_BASE_URL: string = `https://huggingface.co/datalab-to/indra-ocr/resolve/${MODEL_REVISION}`;

/**
 * Mirror of {@link MODEL_BASE_URL} in this repository's models/ folder (git
 * LFS), used only when `PPU_INDRA_OCR_MODEL_MIRROR` is set.
 *
 * It must be the `github.com/<owner>/<repo>/raw/<ref>` form: that one
 * redirects to the LFS media host. `raw.githubusercontent.com` serves the LFS
 * pointer text instead. Off by default: LFS bandwidth has a monthly budget.
 */
export const MODEL_MIRROR_BASE_URL =
  "https://github.com/PT-Perkasa-Pilar-Utama/ppu-indra-ocr/raw/main/models";

/** Default URL of a model file. */
export function defaultModelUrl(role: ModelFile): string {
  return `${MODEL_BASE_URL}/${MODEL_FILES[role]}`;
}

/**
 * Rewrite a built-in model URL onto {@link MODEL_MIRROR_BASE_URL}. Returns
 * null when the mirror is off, or when the URL is not on the primary host: a
 * caller's own model URL is never redirected. The environment is read on
 * every call, so a test can flip it.
 */
export function mirrorUrl(url: string): string | null {
  const enabled = globalThis.process?.env?.PPU_INDRA_OCR_MODEL_MIRROR;
  if (!enabled || enabled === "0" || enabled === "false") return null;
  if (!url.startsWith(MODEL_BASE_URL)) return null;
  return MODEL_MIRROR_BASE_URL + url.slice(MODEL_BASE_URL.length);
}
