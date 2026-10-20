// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/textlayer.py (Apache-2.0, Copyright Datalab).

import type { Gray } from "./image.js";
import type { Page } from "./types.js";
import type { Box } from "./rec-preproc.js";
import { extractRuns, pageTurn, verticalLayout, visualRtlFrac } from "./text-runs.js";
import { roundTo, rot90 } from "./upright.js";

export {
  badCharCount,
  codePoint,
  collapseSpaces,
  extractRuns,
  pageTurn,
  verticalLayout,
  visualRtlFrac,
} from "./text-runs.js";

const MAX_INVISIBLE_FRAC = 0.05;
const MAX_BAD_CHAR_FRAC = 0.005;
const MAX_NO_INK_LINE_FRAC = 0.03;
// Share of neighbouring right-to-left letters in a run that step left to
// right. Logical order steps right to left (sound layers: 0-1%); a layer in
// visual order steps left to right (Chrome-printed Arabic script: 52-94%).
const MAX_VISUAL_RTL_FRAC = 0.25;
const MIN_RTL_PAIRS = 20;
const DARK = 160; // gray level below which a pixel counts as ink
const TEXT_RENDER_INVISIBLE = 3;

/** TextLayer.reason when the layer's character codes are not text. */
export const BROKEN_FONT_MAP = "broken font map";
/** TextLayer.reason when right-to-left letters are stored in visual order. */
export const VISUAL_RTL = "right-to-left text in visual order";

/**
 * One character of a PDF text page: its UTF-16 unit, baseline angle, and box
 * in page space (l, b, r, t), or undefined when pdfium has no box for it.
 */
export type PdfChar = {
  unit: number;
  angle: number;
  box: readonly [number, number, number, number] | undefined;
};

/** What the text-layer code needs from one PDF page. Coordinates are PDF points. */
export type PdfPageSource = {
  /** Page width and height as rendered (crop box and rotation applied). */
  size(): readonly [number, number];
  /** The page rendered in grayscale at `scale` pixels per point. */
  render(scale: number): Gray;
  /** The text page's characters in content order. */
  chars(): PdfChar[];
  /** pdfium's FPDF_PageToDevice on a (width x height) device: integer device x, y. */
  pageToDevice(width: number, height: number, x: number, y: number): readonly [number, number];
  /** Render mode of every text object, nested form objects included. */
  textRenderModes(): number[];
  /** The page's /Rotate in degrees clockwise (0, 90, 180, 270). */
  rotation(): number;
  /** Set the page's /Rotate (degrees clockwise) for the reads that follow. */
  setRotation(degrees: number): void;
};

/**
 * One text-layer run: its box, its text, and its characters with boxes
 * (rendered frame, points), its quarter turns from upright, and its position
 * among all of the page's runs in content order.
 */
export type TextLine = {
  bbox: Box;
  text: string;
  chars: [string, Box][];
  quarter: number;
  seq: number;
};

export type TextLayerStats = {
  chars: number;
  rotated: number;
  bad: number;
  /** Characters at each quarter turn on the rendered page. */
  quarters: [number, number, number, number];
  off_page?: number;
  /** Steps between neighbouring upright characters: down, right. */
  steps?: [number, number];
  no_ink?: number;
  visual_rtl?: number;
  /** The quarter turns the runs were read upright at, and the frame of their boxes. */
  turn?: number;
  frame?: number;
  /** Read as vertical writing: upright characters in a turned frame. */
  vertical?: boolean;
};

/** Upright runs of a page and its character counts. */
export type ExtractedRuns = { runs: TextLine[]; stats: TextLayerStats };

/** The runs of a page and whether to trust them. */
export type TextLayer = {
  /** Upright runs in content order. */
  lines: TextLine[];
  size: readonly [number, number];
  trusted: boolean;
  reason: string;
  stats: TextLayerStats;
  /** Runs at other quarter turns (a sideways table head, a chart axis label). */
  rotated: TextLine[];
};

/** Share of text objects drawn invisibly (an OCR layer over a scan). */
function invisibleTextFrac(page: PdfPageSource): number {
  const modes = page.textRenderModes();
  return modes.length ? modes.filter((m) => m === TEXT_RENDER_INVISIBLE).length / modes.length : 0;
}

/** Share of runs whose box holds no dark pixel in `gray` (`scale` px per point). */
function noInkFrac(gray: Gray, runs: readonly TextLine[], scale: number): number {
  let empty = 0;
  for (const {
    bbox: [x0, y0, x1, y1],
  } of runs) {
    const r0 = Math.max(0, Math.trunc(y0 * scale));
    const r1 = Math.min(gray.height, Math.trunc(y1 * scale) + 1);
    const c0 = Math.max(0, Math.trunc(x0 * scale));
    const c1 = Math.min(gray.width, Math.trunc(x1 * scale) + 1);
    let ink = false;
    for (let y = r0; y < r1 && !ink; y++) {
      for (let x = c0; x < c1; x++) {
        if ((gray.data[y * gray.width + x] ?? 255) < DARK) {
          ink = true;
          break;
        }
      }
    }
    if (!ink) empty++;
  }
  return empty / Math.max(runs.length, 1);
}

const pct = (x: number): string => `${roundTo(x * 100, 0)}%`;

/**
 * Text-layer runs of a page and the verdict on them: invisible text marks an
 * OCR layer, bad characters a broken font map, right-to-left letters in
 * visual order a layer that cannot be put back in reading order, runs with no
 * ink under them hidden or misplaced text. `gray` is a render at `scale`
 * pixels per point, turned like the runs (`frame` quarter turns clockwise).
 */
export function readTextLayer(
  page: PdfPageSource,
  gray: Gray,
  scale: number,
  turn = 0,
  frame: number = turn
): TextLayer {
  const [w, h] = page.size();
  const size: readonly [number, number] = frame % 2 ? [h, w] : [w, h];
  const inv = invisibleTextFrac(page);
  const fail = (
    lines: TextLine[],
    reason: string,
    stats: TextLayerStats,
    rotated: TextLine[] = []
  ): TextLayer => ({ lines, size, trusted: false, reason, stats, rotated });
  if (inv > MAX_INVISIBLE_FRAC) {
    return fail([], `invisible text ${pct(inv)} (OCR layer)`, {
      chars: 0,
      rotated: 0,
      bad: 0,
      quarters: [0, 0, 0, 0],
    });
  }
  const { runs: all, stats } = extractRuns(page, turn, frame);
  stats.turn = turn;
  stats.frame = frame;
  const rotated = all.filter((r) => r.quarter);
  const runs = all.filter((r) => !r.quarter);
  if (!runs.length) return fail([], "no text layer", stats, rotated);
  if (stats.bad > MAX_BAD_CHAR_FRAC * Math.max(stats.chars, 1)) {
    return fail(runs, BROKEN_FONT_MAP, stats, rotated);
  }
  // Only upright text has a reading direction on the page.
  const { pairs, share } = frame === turn ? visualRtlFrac(runs) : { pairs: 0, share: 0 };
  if (pairs >= MIN_RTL_PAIRS && share > MAX_VISUAL_RTL_FRAC) {
    stats.visual_rtl = roundTo(share, 3);
    return fail(runs, VISUAL_RTL, stats, rotated);
  }
  const empty = noInkFrac(gray, runs, scale);
  stats.no_ink = roundTo(empty, 4);
  if (empty > MAX_NO_INK_LINE_FRAC) {
    return fail(runs, `${pct(empty)} of text runs have no ink`, stats, rotated);
  }
  return { lines: runs, size, trusted: true, reason: "", stats, rotated };
}

/** A page's text layer read upright, or the page already read from it. */
export type UprightLayer = { tl: TextLayer; page?: Page };

/**
 * The text layer of a page, read upright (`check` is a render at `scale`
 * pixels per point): a table set sideways is read in the page turned
 * upright, and so is a page whose /Rotate alone turns its text. A trusted
 * layer read without the /Rotate goes to `readUpright` while the page is
 * still turned back; when that returns no page, the turned layer is read
 * instead. Vertical writing is read in the page turned a quarter turn
 * counter-clockwise.
 */
export async function uprightTextLayer(
  page: PdfPageSource,
  check: Gray,
  scale: number,
  mode: string,
  readUpright: (tl: TextLayer) => Promise<Page | undefined>
): Promise<UprightLayer> {
  let tl = readTextLayer(page, check, scale);
  const k = pageTurn(tl.stats);
  const rot = page.rotation();
  if (k && (rot + 90 * k) % 360 === 0) {
    // pdfium builds its text runs in the displayed frame, and on a turned
    // page it can join a line's font runs out of order.
    page.setRotation(0);
    tl = readTextLayer(page, page.render(scale), scale);
    if (tl.lines.length && (tl.trusted || mode === "text")) {
      tl.stats.turn = k;
      const out = await readUpright(tl);
      if (out) return { tl, page: out };
    }
    page.setRotation(rot);
    tl = readTextLayer(page, rot90(check, k), scale, k);
  } else if (k) {
    tl = readTextLayer(page, rot90(check, k), scale, k);
  } else if (verticalLayout(tl.stats)) {
    tl = readTextLayer(page, rot90(check, 3), scale, 0, 3);
    tl.stats.vertical = true;
  }
  return { tl };
}
