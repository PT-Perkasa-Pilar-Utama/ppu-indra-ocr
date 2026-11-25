// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/table_verify.py (Apache-2.0, Copyright Datalab).
//
// The features the false-table verifier reads (table_verify.ts): the table as
// written, the block text, the line geometry, the block size on the page,
// ruling-line pixels in the block crop, and the order model's class
// probabilities.
//
// Numbers follow numpy 2 (NEP 50), as in table-features.ts: float32 arrays
// (boxes, class logits) stay float32 and Python floats are rounded to float32
// before they meet one; results that go through `float()` are float64.

import type { Gray } from "./image.js";
import { meanF32, sumF32 } from "./numeric.js";
import { medianF32 } from "./order-post.js";
import { pyHtmlUnescape } from "./py-text.js";
import { pyStrip } from "./py-unicode.js";
import type { Box } from "./rec-preproc.js";
import { item } from "./util.js";

const f = Math.fround;

/** The line geometry features of `alignment`. */
export type Alignment = {
  g_rows: number;
  g_lines_row: number;
  g_multi_rows: number;
  g_gap: number;
  g_gap_cv: number;
  g_left_edges: number;
  g_right_edges: number;
  g_width_cv: number;
  g_wide_lines: number;
};

/** Feature name to value: what `blockFeatures` computes and the verifier's weights read. */
export type BlockFeatures = Record<string, number>;

const TR = /<tr[^>]*>(.*?)<\/tr>/gs;
const TD = /<t[dh]([^>]*)>(.*?)<\/t[dh]>/gs;
const SPAN = /colspan="(\d+)"/;
const TAG = /<[^>]+>/g;

/** The cell texts of an HTML table, row by row; a cell with colspan n is followed by n - 1 empty cells. */
export function parseTable(html: string): string[][] {
  return [...html.matchAll(TR)].map((row) =>
    [...(row[1] ?? "").matchAll(TD)].flatMap((cell) => {
      const span = Number((SPAN.exec(cell[1] ?? "") ?? [])[1] ?? 1);
      const text = pyStrip(pyHtmlUnescape((cell[2] ?? "").replace(TAG, " ")));
      return [text, ...Array.from({ length: Math.max(span - 1, 0) }, () => "")];
    })
  );
}

/**
 * Per row of `ink` [h, w] (0/1): does a morphological open with a (length x 1)
 * rectangle keep a pixel? The same result as cv2.morphologyEx(MORPH_OPEN) with
 * OpenCV's defaults (anchor at the centre, the border counts as ink for the
 * erosion): a row keeps a pixel when `length` consecutive pixels are ink, when
 * its first b + 1 or last a + 1 pixels are ink (a = length // 2, b = length - 1 -
 * a), or when the whole row is ink.
 */
export function anyRun(ink: Uint8Array, h: number, w: number, length: number): boolean[] {
  const a = length >> 1;
  const b = length - 1 - a;
  return Array.from({ length: h }, (_, y) => {
    const row = ink.subarray(y * w, (y + 1) * w);
    const allInk = (from: number, to: number): boolean => row.subarray(from, to).every(Boolean);
    if (allInk(0, w)) return true;
    if (b + 1 <= w && allInk(0, b + 1)) return true;
    if (a + 1 <= w && allInk(w - a - 1, w)) return true;
    let run = 0;
    for (const v of row) {
      run = v ? run + 1 : 0;
      if (run >= length) return true;
    }
    return false;
  });
}

/** np.percentile(c, 50) truncated to an int, for uint8 values (linear interpolation of the two middle ones). */
function medianInt(values: Uint8Array): number {
  const hist = new Int32Array(256);
  for (const v of values) hist[v] = (hist[v] ?? 0) + 1;
  const kth = (k: number): number => {
    let seen = 0;
    for (let v = 0; v < 256; v++) {
      seen += item(hist, v);
      if (seen > k) return v;
    }
    throw new RangeError("medianInt: empty input");
  };
  const mid = (values.length - 1) / 2;
  return Math.trunc((kth(Math.floor(mid)) + kth(Math.ceil(mid))) / 2);
}

/** Share of the block's height (width) covered by long horizontal (vertical) dark strokes. */
export function ruling(gray: Gray | undefined, box: Box): [number, number] {
  if (!gray) return [0, 0];
  const { width: W, height: H } = gray;
  const x0 = Math.trunc(Math.max(0, f(box[0] - 4)));
  const y0 = Math.trunc(Math.max(0, f(box[1] - 4)));
  const x1 = Math.trunc(Math.min(W, f(box[2] + 4)));
  const y1 = Math.trunc(Math.min(H, f(box[3] + 4)));
  const cw = x1 - x0;
  const ch = y1 - y0;
  if (cw <= 0 || ch <= 0 || Math.min(cw, ch) < 4) return [0, 0];
  const crop = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++)
    crop.set(gray.data.subarray((y0 + y) * W + x0, (y0 + y) * W + x1), y * cw);
  // the threshold stays inside uint8: a negative one would make `<` all false
  const t = Math.max(0, Math.min(160, medianInt(crop) - 30));
  const ink = crop.map((v) => (v < t ? 1 : 0));
  const inkT = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) inkT[x * ch + y] = item(ink, y * cw + x);
  const share = (rows: boolean[]): number => rows.filter(Boolean).length / rows.length;
  return [
    share(anyRun(ink, ch, cw, Math.max(8, Math.floor(cw / 3)))),
    share(anyRun(inkT, cw, ch, Math.max(8, Math.floor(ch / 3)))),
  ];
}

/** numpy's float32 standard deviation: mean, squares and their sum all in float32. */
function stdF32(values: readonly number[]): number {
  const m = meanF32(values);
  const sq = values.map((v) => f(f(v - m) * f(v - m)));
  return f(Math.sqrt(f(sumF32(sq) / values.length)));
}

/** `std(values) / max(1e-6, mean(scale))` in float32, as the reference writes it. */
export function spreadF32(values: readonly number[], scale: readonly number[] = values): number {
  const m = meanF32(scale);
  return f(stdF32(values) / f(m > 1e-6 ? m : 1e-6));
}

/** Distinct edges: values further apart than `gap` start a new edge. */
function edges(values: readonly number[], gap: number): number {
  const v = values.toSorted((p, q) => p - q);
  return values.length ? 1 + v.slice(1).filter((e, i) => f(e - item(v, i)) > gap).length : 0;
}

/**
 * Line geometry inside the block: rows by vertical centre (overlap > half a line
 * height), lines per row, the gap between neighbours in a row, and how many
 * distinct left / right edges the lines use (within half a line height).
 * `b` holds the block's boxes as float32 values.
 */
export function alignment(b: readonly Box[]): Alignment {
  const h = medianF32(b.map(([, y0, , y1]) => f(y1 - y0))) || 1;
  const cy = b.map(([, y0, , y1]) => f(f(y0 + y1) / 2));
  const half = f(0.5 * h);
  const rows: number[][] = [];
  for (const i of cy.map((_, k) => k).toSorted((p, q) => item(cy, p) - item(cy, q))) {
    const last = rows.at(-1);
    if (last && Math.abs(f(item(cy, i) - meanF32(last.map((j) => item(cy, j))))) < half)
      last.push(i);
    else rows.push([i]);
  }
  const gaps = rows.flatMap((r) => {
    const xs = r.toSorted((p, q) => item(b, p)[0] - item(b, q)[0]);
    return xs.slice(1).map((k, i) => f(f(item(b, k)[0] - item(b, item(xs, i))[2]) / f(h)));
  });
  const n = b.length;
  const widths = b.map(([x0, , x1]) => f(x1 - x0));
  const span = f(f(0.6) * f(Math.max(...b.map((x) => x[2])) - Math.min(...b.map((x) => x[0]))));
  return {
    g_rows: Math.log1p(rows.length),
    g_lines_row: n / Math.max(1, rows.length),
    g_multi_rows: rows.filter((r) => r.length >= 2).length / Math.max(1, rows.length),
    g_gap: gaps.length ? Math.log1p(Math.max(0, medianF32(gaps))) : 0,
    g_gap_cv: gaps.length > 1 ? spreadF32(gaps, gaps.map(Math.abs)) : 0,
    g_left_edges:
      edges(
        b.map((x) => x[0]),
        half
      ) / Math.max(1, n),
    g_right_edges:
      edges(
        b.map((x) => x[2]),
        half
      ) / Math.max(1, n),
    g_width_cv: spreadF32(widths),
    g_wide_lines: widths.filter((w) => w > span).length / n,
  };
}
