// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession, Tensor } from "onnxruntime-common";

import type { DetBox } from "./det-post.js";
import type { Gray } from "./image.js";
import type { LayoutBlock } from "./layout-post.js";
import type { Box } from "./rec-preproc.js";

/**
 * One recognized character position: a class id (Latin recognizer) or a
 * character (CJK recognizer, PDF text layer, math), its page x, and for text
 * layer characters their index in the PDF content stream.
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

/** orient.onnx's verdict: quarter turns (counter-clockwise) the content is turned by, and the class probabilities. */
export type Orientation = { turns: number; probs: number[] };

/** A page turned upright and deskewed, with the turns and degrees undone. */
export type Straightened = { image: Gray; turns: number; skew: number };

/** The `meta.json` that ships with the models. */
export type IndraMeta = {
  vocab: string;
  vocab_cjk?: string;
  rec_height: number;
  det_long_side: number;
};

/** ONNX Runtime sessions of the Indra models. Optional ones switch a stage off. */
export type IndraSessions = {
  det: InferenceSession;
  rec: InferenceSession;
  order?: InferenceSession;
  orient?: InferenceSession;
  script?: InferenceSession;
  recCjk?: InferenceSession;
};

export type TensorCtor = typeof Tensor;
