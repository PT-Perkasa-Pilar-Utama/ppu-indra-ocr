// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession, Tensor } from "onnxruntime-common";

import type { DetBox } from "./det-post.js";
import type { Gray } from "./image.js";
import type { LayoutBlock } from "./layout-post.js";
import type { NgramLM } from "./lm.js";
import type { MathReader } from "./math.js";
import type { Box } from "./rec-preproc.js";

/**
 * One recognized character position: a class id (main recognizer) or a
 * character (group recognizer, PDF text layer, math), its page x, and for
 * text-layer characters their index in the PDF content stream.
 */
export type Tok = readonly [id: number | string, x: number, contentIndex?: number];

/** Text, confidence and character positions of one detected line. */
export type RecLine = { text: string; conf: number; toks: readonly Tok[] };

/** One output line, in reading order. */
export type Line = { box: Box; text: string; conf: number };

/** Where a page's text came from. */
export type PageSource = "ocr" | "text" | "hybrid" | "error";

/** The result for one page. Boxes are in the upright page (after any quarter turn). */
export type Page = {
  /** Lines in reading order. */
  lines: Line[];
  /** All lines, one per row, in reading order. */
  text: string;
  /** Headings, paragraphs, lists, and tables as HTML. Page headers and footers are left out by default. */
  markdown: string | null;
  /** Layout blocks in reading order. */
  blocks: LayoutBlock[];
  source: PageSource;
  /** Why the page took its path. */
  note: string;
  /** Degrees counter-clockwise the input was turned. */
  rotation: number;
  /** Degrees of small tilt undone; boxes are mapped back to the turned page. */
  skew: number;
  /**
   * Set in vertical columns (Chinese, Japanese): read turned a quarter turn,
   * columns as lines. Boxes are on the page as it is (rotation stays 0).
   */
  vertical: boolean;
  /** Milliseconds per stage, with total_ms. */
  timings: Record<string, number>;
};

/** Reading order, and the layout head outputs when the order model has them. */
export type ReadingOrder = {
  order: number[];
  cls?: Float32Array;
  same?: Float32Array;
  logits?: Float32Array;
};

/** A piece of a split line: its box, text and tokens. */
export type LinePart = { box: Box; text: string; toks: readonly Tok[] };

/** One detector pass: boxes in detector-input pixels, and the page scale. */
export type DetectPass = { boxes: DetBox[]; scale: number };

/**
 * orient.onnx's verdict: quarter turns (counter-clockwise) the content is
 * turned by, or VERTICAL (4) for a page set in vertical columns, and the
 * class probabilities.
 */
export type Orientation = { turns: number; probs: number[] };

/** A page turned upright and deskewed, with the turns and degrees undone. */
export type Straightened = { image: Gray; turns: number; skew: number };

/** Per-group language model settings in meta.json (`lm_params`). */
export type LmParams = { alpha?: number; beta?: number; gate?: number };

/** The `meta.json` that ships with the models. */
export type IndraMeta = {
  vocab: string;
  rec_height: number;
  det_long_side: number;
  /** rec.onnx reads a window this many line heights tall (the line and the page around it). */
  rec_window?: number;
  /** script.onnx's class order (default base, cjk, rtl, other). */
  script_groups?: string[];
  lm_params?: Record<string, LmParams>;
} & { [group: `vocab_${string}`]: string | undefined };

/** A model that loads on its first use. */
export type Lazy<T> = () => Promise<T>;

/**
 * ONNX Runtime sessions of the Indra models. Optional ones switch a stage
 * off. `groups` holds the recognizer of each script group (cjk, rtl, world),
 * `lms` the language model of a recognizer ("base" is rec.onnx); both load on
 * the first line that needs them.
 */
export type IndraSessions = {
  det: InferenceSession;
  rec: InferenceSession;
  order?: InferenceSession;
  orient?: InferenceSession;
  script?: InferenceSession;
  groups?: Record<string, Lazy<InferenceSession>>;
  lms?: Record<string, Lazy<NgramLM>>;
};

export type TensorCtor = typeof Tensor;

/** Pipeline settings beyond the models. */
export type PipelineOptions = {
  /** Keep page headers and footers in the markdown. Default false. */
  furniture?: boolean;
  /** Run a second detection pass on pages with small text. Default true. */
  adaptive?: boolean;
  /** Read lines with math as LaTeX. Off unless given. */
  math?: MathReader;
  /** Re-read unsure lines with the language models (when given). Default true. */
  lm?: boolean;
};
