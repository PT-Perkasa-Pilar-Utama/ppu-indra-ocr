// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import type { InferenceSession } from "onnxruntime-common";

import type { ModelFile, ModelPreset } from "./model-catalogue.js";

/**
 * The image engine for resizing and warping.
 *
 * - `"opencv"` - OpenCV.js through ppu-ocv: the same OpenCV release as the
 *   Python reference, so results match it. **(default on Node and Bun)**
 * - `"canvas-native"` - plain TypeScript, no OpenCV: smaller and runs
 *   anywhere, results within a token or two of the reference.
 */
export type ProcessingEngine = "opencv" | "canvas-native";

/**
 * Where to read each model file from: a URL, a file path (Node), or the bytes.
 * Files left out come from the default model repository.
 */
export type ModelSources = Partial<Record<ModelFile, string | ArrayBuffer>>;

/** ONNX Runtime session options, plus a hook for a silent provider fallback. */
export type SessionOptions = InferenceSession.SessionOptions & {
  /**
   * Execution providers to use (for example "cpu", "cuda", "webgpu", "wasm").
   * @default ['cpu']
   */
  executionProviders?: InferenceSession.SessionOptions["executionProviders"];

  /**
   * Called when session creation with the requested execution providers fails
   * and the service retries with a safe provider (`cpu` or `wasm`). Receives
   * the original error. Never forwarded to ONNX Runtime.
   */
  onSessionFallback?: (error: unknown) => void;
};

/** Options of an IndraOcrService. */
export type IndraOptions = {
  /**
   * The model set: "v1.1" (Indra OCR 1.1, the default) or "v3.1" (the first
   * release, smaller and Latin, Cyrillic, Greek and Chinese only).
   */
  model?: ModelPreset;
  /** Per-file model sources. Default: Datalab's repository at the preset's pinned revision. */
  models?: ModelSources;
  /**
   * Re-read lines the recognizer is unsure of with the n-gram language models
   * (v1.1). Each model loads on the first such line: 16 to 19 MB each.
   * Default true on Node and Bun, false on web and mobile.
   */
  lm?: boolean;
  /** Read lines with math as LaTeX. Loads four more models. Default false. */
  math?: boolean;
  /** Keep page headers and footers in the markdown. Default false. */
  furniture?: boolean;
  /** Run a second detection pass on pages with small text. Default true. */
  adaptive?: boolean;
  /** Image engine. Default "opencv" on Node and Bun, "canvas-native" on web and mobile. */
  engine?: ProcessingEngine;
  /** ONNX Runtime session options for every model. */
  session?: SessionOptions;
  /** Log loading and stage steps to the console. Default false. */
  verbose?: boolean;
};

/** Options of convertBatch and convertStream. */
export type BatchOptions = {
  /**
   * Pages in flight at once. Default 1: each page allocates large tensors,
   * so more pages at once mostly add memory.
   */
  concurrency?: number;
  /** Report a failed image as a rejected item instead of failing the whole batch. Default false. */
  settle?: boolean;
  /** Stops scheduling further images; the batch rejects with an AbortError. */
  signal?: AbortSignal;
  /** Called after each image settles, with the done count and the total when known. */
  onProgress?: (done: number, total: number | undefined) => void;
};
