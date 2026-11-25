// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession, Tensor } from "onnxruntime-common";

import type { DetBox } from "./det-post.js";
import type { Gray } from "./image.js";
import type { LayoutBlock } from "./layout-post.js";
import type { NgramLM } from "./lm.js";
import type { TsrSessions } from "./table-rel.js";
import type { VerifyFile } from "./table-verify.js";
import type { MathReader } from "./math.js";
import type { Box } from "./rec-preproc.js";

/**
 * What a token added by an inline formula splice (math-span-img.ts) stands
 * for: the formula, with the recognizer's tokens it replaced, or padding the
 * splice added (a space around the formula, an empty token at a replaced
 * character's x).
 */
export type Splice = { kind: "formula"; held: readonly Tok[] } | { kind: "pad" };

/**
 * One recognized character position: a class id (main recognizer) or a
 * character (group recognizer, PDF text layer, math), its page x, for
 * text-layer characters their index in the PDF content stream, and for a
 * token an inline formula splice added, what it stands for.
 */
export type Tok = readonly [id: number | string, x: number, contentIndex?: number, splice?: Splice];

/**
 * A math reading that replaced a line, kept for the layout's second look:
 * the accept head also weighs whether the line sits in a table, which only
 * the reading order shows. `plain` is the line as read before.
 */
export type MathAccepted = {
  plain: RecLine;
  reading: string;
  conf: number;
  draft: string;
  rconf: number;
};

/** Text, confidence and character positions of one detected line; `math` when a math reading replaced it. */
export type RecLine = { text: string; conf: number; toks: readonly Tok[]; math?: MathAccepted };

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

/**
 * One detector pass: boxes in detector-input pixels, the page scale, and the
 * text logit map at OUT_STRIDE of the detector input (tiled detection reads
 * where the page holds text from it).
 */
export type DetectPass = {
  boxes: DetBox[];
  scale: number;
  logit: { data: Float32Array; width: number; height: number };
};

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
  /** rec_hw.onnx's input height and window (default: rec.onnx's). */
  rec_hw_height?: number;
  rec_hw_window?: number;
  /** A base line goes to rec_hw.onnx at this hw.onnx probability or more. */
  hw_route_min?: number;
  /** order.onnx line thumbnails: resize mode (default area). */
  order_thumbs_interp?: "area" | "linear";
  /** false: PDF text-layer tables keep the geometric assembly. */
  tsr_rel_layer?: boolean;
  /** How a table cell's column and row spans are found ("geo", "rel", "rel2"). */
  tsr_rel_span?: string;
} & { [group: `vocab_${string}`]: string | undefined } & {
  /** A group recognizer that reads lines widened by this factor. */
  [group: `rec_xscale_${string}`]: number | undefined;
};

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
  /**
   * Handwriting: hw.onnx (printed or handwritten, on the line band) and
   * rec_hw.onnx, which reads the base lines hw.onnx calls handwritten. The
   * recognizer loads on the first such line.
   */
  hw?: { cls: InferenceSession; rec: Lazy<InferenceSession> };
  /** Table structure: tsr_rel.onnx with tsr_split and tsr_refine, loaded on the first table. */
  tables?: Lazy<TsrSessions>;
  /** The false-table verifier's weights (table_verify.json). */
  tableVerify?: VerifyFile;
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
