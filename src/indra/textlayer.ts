// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/textlayer.py (Apache-2.0, Copyright Datalab).

import { pyRound } from "./det-post.js";
import type { Gray } from "./image.js";
import { medianF32 } from "./order-post.js";
import type { Box } from "./rec-preproc.js";
import { roundTo } from "./upright.js";
import { isPySpace, item } from "./util.js";

const MAX_INVISIBLE_FRAC = 0.05;
const MAX_BAD_CHAR_FRAC = 0.005;
const MAX_NO_INK_LINE_FRAC = 0.03;
const DARK = 160; // gray level below which a pixel counts as ink
const MIN_MODEL_AGREEMENT = 0.85;
const TEXT_RENDER_INVISIBLE = 3;

/** TextLayer.reason when the layer's character codes are not text. */
export const BROKEN_FONT_MAP = "broken font map";

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
};

/** One text-layer run: its box, its text, and its characters with boxes (rendered frame, points). */
export type TextLine = { bbox: Box; text: string; chars: [string, Box][] };

export type TextLayerStats = {
  chars: number;
  rotated: number;
  bad: number;
  off_page?: number;
  no_ink?: number;
};

/** Upright runs of a page and its character counts. */
export type ExtractedRuns = { runs: TextLine[]; stats: TextLayerStats };

/** The runs of a page and whether to trust them. */
export type TextLayer = {
  lines: TextLine[];
  size: readonly [number, number];
  trusted: boolean;
  reason: string;
  stats: TextLayerStats;
};

const BAD = /^[\p{Co}\p{Cs}\p{Cn}]$/u;
const CONTROL = /^\p{Cc}$/u;

/** Replacement, private-use, surrogate, unassigned and control characters (tab excepted). */
export function badCharCount(text: string): number {
  let n = 0;
  // One UTF-16 unit at a time: a lone surrogate from the text page counts as bad (category Cs).
  for (let i = 0; i < text.length; i++) {
    const c = text.charAt(i);
    if (c.charCodeAt(0) === 0xfffd || BAD.test(c) || (CONTROL.test(c) && c !== "\t")) n++;
  }
  return n;
}

/** [2, 3] map from PDF page space to the rendered frame (points, top-left origin), from FPDF_PageToDevice. */
function deviceAffine(page: PdfPageSource): number[] {
  const [w, h] = page.size();
  const S = 1000; // device units per point: keeps the integer output precise
  const dw = pyRound(w * S);
  const dh = pyRound(h * S);
  const dev = (x: number, y: number): [number, number] => {
    const [dx, dy] = page.pageToDevice(dw, dh, x, y);
    return [dx / S, dy / S];
  };
  const [o0, o1] = dev(0, 0);
  const [ex0, ex1] = dev(1, 0);
  const [ey0, ey1] = dev(0, 1);
  return [ex0 - o0, ey0 - o0, o0, ex1 - o1, ey1 - o1, o1];
}

/** Upright text runs of a page, in content order, and character counts. */
export function extractRuns(page: PdfPageSource): ExtractedRuns {
  const [a00 = 0, a01 = 0, a02 = 0, a10 = 0, a11 = 0, a12 = 0] = deviceAffine(page);
  const upright = new Map<number, boolean>();
  const chars: string[] = [];
  const raw: (readonly [number, number, number, number])[] = [];
  const breaks: number[] = []; // runs end after these positions
  const stats: TextLayerStats = { chars: 0, rotated: 0, bad: 0 };
  for (const ch of page.chars()) {
    let c = String.fromCharCode(ch.unit);
    if (c === "\r" || c === "\n") {
      breaks.push(chars.length);
      continue;
    }
    if (c === "\u0002") c = "-"; // pdfium's mark for a hyphen at a line break
    if (!isPySpace(c)) {
      stats.chars++;
      stats.bad += badCharCount(c);
      let up = upright.get(ch.angle);
      if (up === undefined) {
        // Upright on the rendered page: the baseline, mapped through the
        // page-to-render transform, runs left to right.
        const d0 = a00 * Math.cos(ch.angle) + a01 * Math.sin(ch.angle);
        const d1 = a10 * Math.cos(ch.angle) + a11 * Math.sin(ch.angle);
        up = !(d0 <= 0 || Math.abs(d1) > 1e-3 * Math.max(Math.abs(d0), 1e-9));
        upright.set(ch.angle, up);
      }
      if (!up) {
        stats.rotated++;
        breaks.push(chars.length);
        continue;
      }
    }
    if (!ch.box) throw new Error("Failed to get charbox.");
    chars.push(c);
    raw.push(ch.box);
  }
  if (!chars.length) return { runs: [], stats };
  const [w, h] = page.size();
  const boxes = raw.map(([l, b, r, t]): Box => {
    const px = [l, r, l, r].map((x, i) => a00 * x + a01 * item([b, b, t, t], i) + a02);
    const py = [l, r, l, r].map((x, i) => a10 * x + a11 * item([b, b, t, t], i) + a12);
    return [Math.min(...px), Math.min(...py), Math.max(...px), Math.max(...py)];
  });
  // Characters outside the rendered frame (beyond a crop box) are not on the page.
  const visible = boxes.map(([x0, y0, x1, y1]) => {
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    return cx >= 0 && cx <= w && cy >= 0 && cy <= h;
  });
  stats.off_page = visible.filter((v) => !v).length;
  const runs: TextLine[] = [];
  let start = 0;
  for (const end of [...new Set([...breaks, chars.length])].toSorted((p, q) => p - q)) {
    const seg: [string, Box][] = [];
    for (let i = start; i < end; i++) if (visible[i]) seg.push([item(chars, i), item(boxes, i)]);
    start = end;
    const ink = seg.filter(([c]) => !isPySpace(c)).map(([, b]) => b);
    if (!ink.length) continue;
    runs.push({
      bbox: [
        Math.min(...ink.map((b) => b[0])),
        Math.min(...ink.map((b) => b[1])),
        Math.max(...ink.map((b) => b[2])),
        Math.max(...ink.map((b) => b[3])),
      ],
      text: collapseSpaces(seg.map(([c]) => c).join("")),
      chars: seg,
    });
  }
  return { runs, stats };
}

/** Python's " ".join(s.split()). */
export function collapseSpaces(s: string): string {
  return [...s]
    .reduce<string[]>(
      (words, c) => {
        if (isPySpace(c)) words.push("");
        else words[words.length - 1] += c;
        return words;
      },
      [""]
    )
    .filter(Boolean)
    .join(" ");
}

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
 * OCR layer, bad characters a broken font map, runs with no ink under them
 * hidden or misplaced text. `gray` is a render at `scale` pixels per point.
 */
export function readTextLayer(page: PdfPageSource, gray: Gray, scale: number): TextLayer {
  const size = page.size();
  const inv = invisibleTextFrac(page);
  const fail = (lines: TextLine[], reason: string, stats: TextLayerStats): TextLayer => ({
    lines,
    size,
    trusted: false,
    reason,
    stats,
  });
  if (inv > MAX_INVISIBLE_FRAC) {
    return fail([], `invisible text ${pct(inv)} (OCR layer)`, { chars: 0, rotated: 0, bad: 0 });
  }
  const { runs, stats } = extractRuns(page);
  if (!runs.length) return fail([], "no text layer", stats);
  if (stats.bad > MAX_BAD_CHAR_FRAC * Math.max(stats.chars, 1))
    return fail(runs, BROKEN_FONT_MAP, stats);
  const empty = noInkFrac(gray, runs, scale);
  stats.no_ink = roundTo(empty, 4);
  if (empty > MAX_NO_INK_LINE_FRAC)
    return fail(runs, `${pct(empty)} of text runs have no ink`, stats);
  return { lines: runs, size, trusted: true, reason: "", stats };
}

/** Upward moves in content order that do not go to a column on the right. */
export function orderBacktracks(boxes: readonly Box[]): number {
  if (boxes.length < 3) return 0;
  const f = Math.fround;
  const h = medianF32(boxes.map(([, y0, , y1]) => f(y1 - y0)));
  let bt = 0;
  for (let i = 1; i < boxes.length; i++) {
    const a = item(boxes, i - 1);
    const b = item(boxes, i);
    if (b[1] < f(a[1] - f(0.5 * h)) && b[0] < f(a[2] - f(h))) bt++;
  }
  return bt;
}

/**
 * Is the content order a reading order? Few upward moves that are not column
 * changes, and the order model (pairwise logits [n, n], a[i, j] > 0: i
 * before j) agrees with most consecutive pairs.
 */
export function pdfOrderCoherent(boxes: readonly Box[], pairLogits?: Float32Array): boolean {
  const n = boxes.length;
  if (orderBacktracks(boxes) > Math.max(1, Math.trunc(0.02 * n))) return false;
  if (!pairLogits || n < 3) return true;
  let agree = 0;
  for (let i = 0; i < n - 1; i++) if ((pairLogits[i * n + i + 1] ?? 0) > 0) agree++;
  return agree / (n - 1) >= MIN_MODEL_AGREEMENT;
}
