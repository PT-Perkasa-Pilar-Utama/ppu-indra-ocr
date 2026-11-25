// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/table_rel.py (Apache-2.0, Copyright Datalab).
//
// The words of a table block and their features for tsr_rel.onnx.
//
// Numbers follow numpy 2 (NEP 50): a float32 array combined with a Python float
// stays float32 and the float is rounded first, so each such step is
// `Math.fround(a op Math.fround(c))`. Python floats stay float64.

import type { Gray } from "./image.js";
import { medianF32 } from "./order-post.js";
import { LETTER, UPPER } from "./py-text.js";
import { isDigit, pyLen, pyStrip } from "./py-unicode.js";
import type { Box } from "./rec-preproc.js";
import type { Tok } from "./types.js";
import { item } from "./util.js";

const f = Math.fround;

export const N_FEAT_BASE = 33; // 26 word features + 7 ruling-line extents (a model that takes 26 reads the first 26)

const PUNCT = /^\p{P}$/u;
const SYMBOL = /^\p{S}$/u;

/** A word of a table line. `pos`: index in its line, `n`: words in its line. */
export type Word = { box: Box; text: string; line: number; pos: number; n: number };

/** A line of a table block: its box on the page and its tokens. */
export type TableLine = { box: Box; toks: readonly Tok[] };

/** What the recognizer's vocabulary says about tokens. */
export type WordText = {
  isSpace(id: Tok[0]): boolean;
  decode(toks: readonly Tok[]): string;
};

/** A table's box on its page: x0, y0, x1, y1. */
export type TableBox = readonly [number, number, number, number];

/** Words from lines, cut at the space tokens. A word's box spans its tokens' x positions at the line's height. */
export function wordsOf(lines: readonly TableLine[], text: WordText): Word[] {
  const out: Word[] = [];
  for (const [li, { box, toks }] of lines.entries()) {
    const h = Math.max(1, f(box[3] - box[1]));
    const words: Tok[][] = [];
    let cur: Tok[] = [];
    for (const tk of toks) {
      if (text.isSpace(tk[0])) {
        if (cur.length) words.push(cur);
        cur = [];
      } else {
        cur.push(tk);
      }
    }
    if (cur.length) words.push(cur);
    const ws: Word[] = [];
    for (const w of words) {
      const xs = w.map((t) => t[1]);
      const t = pyStrip(text.decode(w));
      if (!t) continue;
      let x0 = Math.max(box[0], Math.min(...xs) - 0.35 * h);
      let x1 = Math.min(box[2], Math.max(...xs) + 0.35 * h);
      if (x1 <= x0) {
        x0 = Math.min(...xs) - 0.3 * h;
        x1 = Math.max(...xs) + 0.3 * h;
      }
      ws.push({ box: [x0, box[1], x1, box[3]], text: t, line: li, pos: 0, n: 0 });
    }
    for (const [k, w] of ws.entries()) {
      w.pos = k;
      w.n = ws.length;
    }
    out.push(...ws);
  }
  return out;
}

/** The words' boxes as a float32 [n, 4] array, as the reference stores them. */
export function boxesF32(words: readonly Word[]): Float32Array {
  const out = new Float32Array(words.length * 4);
  for (const [i, w] of words.entries()) out.set(w.box, i * 4);
  return out;
}

/** The union of float32 [n, 4] boxes. */
export function unionBox(boxes: Float32Array): TableBox {
  let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let i = 0; i < boxes.length; i += 4) {
    x0 = Math.min(x0, item(boxes, i));
    y0 = Math.min(y0, item(boxes, i + 1));
    x1 = Math.max(x1, item(boxes, i + 2));
    y1 = Math.max(y1, item(boxes, i + 3));
  }
  return [x0, y0, x1, y1];
}

/** Shares of digits, letters, capitals, punctuation and symbols, then 1 when any character is of a wide script, for one text. */
function textFeatures(t: string): number[] {
  const tot = [0, 0, 0, 0, 0, 0];
  const bump = (k: number, on: boolean): void => {
    tot[k] = (tot[k] ?? 0) + (on ? 1 : 0);
  };
  for (const c of t || " ") {
    bump(0, isDigit(c));
    bump(1, LETTER.test(c));
    bump(2, UPPER.test(c));
    bump(3, PUNCT.test(c));
    bump(4, SYMBOL.test(c));
    bump(5, (c.codePointAt(0) ?? 0) > 0x2e7f);
  }
  const n = Math.max(pyLen(t), 1);
  return [...tot.slice(0, 5).map((v) => v / n), tot[5] ? 1 : 0];
}

/**
 * The horizontal ink rule in rows [y0, y1) (page px) that covers at least 70%
 * of the word's span [x0, x1): its ink run's (left, right) in page px, or null.
 * `ink` is the 0/1 crop of the page from (ox, oy), rw x rh.
 */
function rule(
  ink: Uint8Array,
  [rw, rh, ox, oy]: readonly [number, number, number, number],
  [x0, x1, y0, y1]: readonly [number, number, number, number]
): [number, number] | null {
  const r0 = Math.max(Math.trunc(y0) - oy, 0);
  const r1 = Math.min(Math.ceil(y1) - oy, rh);
  const c0 = Math.max(Math.trunc(x0) - ox, 0);
  const c1 = Math.min(Math.ceil(x1) - ox, rw);
  if (r1 <= r0 || c1 - c0 < 2) return null;
  let j = 0;
  let best = -1;
  for (let r = r0; r < r1; r++) {
    let sum = 0;
    for (let c = c0; c < c1; c++) sum += item(ink, r * rw + c);
    if (sum / (c1 - c0) > best) {
      best = sum / (c1 - c0);
      j = r - r0;
    }
  }
  if (best < 0.7) return null;
  // A 3-row band: rules can be 1-2 px thick or slanted.
  const row = new Uint8Array(rw);
  for (let r = Math.max(r0 + j - 1, 0); r < Math.min(r0 + j + 2, rh); r++)
    for (let c = 0; c < rw; c++) if (item(ink, r * rw + c)) row[c] = 1;
  const mid = (c0 + c1) >> 1;
  if (!row[mid]) return null;
  let a = mid;
  while (a > 0 && row[a - 1]) a--;
  let b = mid + 1;
  while (b < rw && row[b]) b++;
  return [a + ox, b + ox];
}

/** [N, N_FEAT_BASE] float32 features for the words of one table; `bbox` is the table's box on `gray`, the page. */
export function wordFeatures(words: readonly Word[], gray: Gray, bbox: TableBox): Float32Array {
  const n = words.length;
  const out = new Float32Array(n * N_FEAT_BASE);
  if (n === 0) return out;
  const boxes = boxesF32(words);
  const X0 = (i: number): number => item(boxes, i * 4);
  const Y0 = (i: number): number => item(boxes, i * 4 + 1);
  const X1 = (i: number): number => item(boxes, i * 4 + 2);
  const Y1 = (i: number): number => item(boxes, i * 4 + 3);
  const rawH = Array.from({ length: n }, (_, i) => f(Y1(i) - Y0(i)));
  const hs = rawH.map((h) => Math.max(h, 1));
  const hm = medianF32(rawH);
  const [tx0, ty0, tx1, ty1] = bbox;
  const tw = Math.max(tx1 - tx0, 1);
  const th = Math.max(ty1 - ty0, 1);
  const set = (i: number, col: number, v: number): void => {
    out[i * N_FEAT_BASE + col] = v;
  };
  for (const [i, w] of words.entries()) {
    set(i, 0, f(f(X0(i) - f(tx0)) / f(tw)));
    set(i, 1, f(f(Y0(i) - f(ty0)) / f(th)));
    set(i, 2, f(f(X1(i) - f(tx0)) / f(tw)));
    set(i, 3, f(f(Y1(i) - f(ty0)) / f(th)));
    set(i, 4, f(f(f(X0(i) - f(tx0)) / hm) / 50));
    set(i, 5, f(f(f(Y0(i) - f(ty0)) / hm) / 50));
    set(i, 6, f(f(f(X1(i) - X0(i)) / hm) / 20));
    set(i, 7, f(item(hs, i) / hm));
    const t = w.text;
    set(i, 8, Math.log1p(pyLen(t)) / 4);
    const first = [...t][0];
    set(i, 14, first !== undefined && isDigit(first) ? 1 : 0);
    set(i, 15, t.endsWith(":") ? 1 : 0);
    set(i, 16, w.pos / Math.max(w.n - 1, 1));
    set(i, 17, w.pos === 0 ? 1 : 0);
    set(i, 18, w.pos === w.n - 1 ? 1 : 0);
    const tf = textFeatures(t);
    for (let k = 0; k < 5; k++) set(i, 9 + k, item(tf, k));
    set(i, 25, item(tf, 5));
  }
  const byLine = new Map<number, number[]>();
  for (const [i, w] of words.entries()) byLine.set(w.line, [...(byLine.get(w.line) ?? []), i]);
  for (const line of byLine.values()) {
    for (const [k, i] of line.entries()) {
      const h = item(hs, i);
      set(i, 19, k > 0 ? Math.min(f(f(X0(i) - X1(item(line, k - 1))) / h), 10) / 10 : 1);
      set(
        i,
        20,
        k + 1 < line.length ? Math.min(f(f(X0(item(line, k + 1)) - X1(i)) / h), 10) / 10 : 1
      );
    }
  }
  addRulingFeatures(out, { boxes, hs, gray, bbox, tw });
  return out;
}

type RulingInput = {
  boxes: Float32Array;
  hs: readonly number[];
  gray: Gray;
  bbox: TableBox;
  tw: number;
};

/**
 * Ink in thin strips just outside each word (ruling lines, cell borders), from
 * an integral image of the table's region of the page, and the rules over and
 * under each word. A horizontal rule right under (over) a word, and how far it
 * reaches past the word to the left and right: a group header's rule spans the
 * columns it heads. [26] below, [27] its left reach, [28] its right reach (table
 * widths); [29..31] the same above; [32] = 1: these features are present.
 */
function addRulingFeatures(out: Float32Array, { boxes, hs, gray, bbox, tw }: RulingInput): void {
  const [tx0, ty0, tx1, ty1] = bbox;
  const { width: W, height: H } = gray;
  const m = Math.max(...hs); // the strips reach 0.5 h beyond a word
  const ox = Math.max(Math.trunc(tx0 - m), 0);
  const oy = Math.max(Math.trunc(ty0 - m), 0);
  const rw = Math.min(Math.ceil(tx1 + m), W) - ox;
  const rh = Math.min(Math.ceil(ty1 + m), H) - oy;
  if (rw <= 0 || rh <= 0) return;
  const ink = new Uint8Array(rw * rh);
  const ii = new Int32Array((rh + 1) * (rw + 1));
  for (let y = 0; y < rh; y++) {
    let run = 0;
    for (let x = 0; x < rw; x++) {
      const dark = item(gray.data, (oy + y) * W + ox + x) < 128 ? 1 : 0;
      ink[y * rw + x] = dark;
      run += dark;
      ii[(y + 1) * (rw + 1) + x + 1] = item(ii, y * (rw + 1) + x + 1) + run;
    }
  }
  const clip = (v: number, hi: number): number => Math.min(Math.max(v, 0), hi);
  const strip = (a0: number, b0: number, a1: number, b1: number): number => {
    const c0 = clip(Math.floor(f(a0 - ox)), rw);
    const r0 = clip(Math.floor(f(b0 - oy)), rh);
    const c1 = clip(Math.ceil(f(a1 - ox)), rw);
    const r1 = clip(Math.ceil(f(b1 - oy)), rh);
    const area = (c1 - c0) * (r1 - r0);
    const at = (r: number, c: number): number => item(ii, r * (rw + 1) + c);
    return area > 0 ? (at(r1, c1) - at(r0, c1) - at(r1, c0) + at(r0, c0)) / area : 0;
  };
  const k = (c: number, h: number): number => f(f(c) * h); // `c * h`, h a float32 array
  for (const [i, h] of hs.entries()) {
    const x0 = item(boxes, i * 4);
    const y0 = item(boxes, i * 4 + 1);
    const x1 = item(boxes, i * 4 + 2);
    const y1 = item(boxes, i * 4 + 3);
    const set = (col: number, v: number): void => {
      out[i * N_FEAT_BASE + col] = v;
    };
    set(21, strip(x0, f(y0 - k(0.45, h)), x1, f(y0 - k(0.1, h))));
    set(22, strip(x0, f(y1 + k(0.1, h)), x1, f(y1 + k(0.45, h))));
    set(23, strip(f(x0 - k(0.5, h)), y0, f(x0 - k(0.1, h)), y1));
    set(24, strip(f(x1 + k(0.1, h)), y0, f(x1 + k(0.5, h)), y1));
    set(32, 1);
    const spans: [number, number][] = [
      [f(y1 + k(0.05, h)), f(y1 + k(0.8, h))],
      [f(y0 - k(0.8, h)), f(y0 - k(0.05, h))],
    ];
    for (const [side, [lo, hi]] of spans.entries()) {
      const r = rule(ink, [rw, rh, ox, oy], [x0, x1, lo, hi]);
      if (r) {
        set(26 + 3 * side, 1);
        set(27 + 3 * side, Math.min((x0 - r[0]) / tw, 1));
        set(28 + 3 * side, Math.min((r[1] - x1) / tw, 1));
      }
    }
  }
}

/** Word boxes (float32 [N, 4], px) in median line heights from the top-left word corner: the model's box input. */
export function normBoxes(boxes: Float32Array): Float32Array {
  const n = boxes.length / 4;
  if (n === 0) return new Float32Array(0);
  const heights = Array.from({ length: n }, (_, i) =>
    f(item(boxes, i * 4 + 3) - item(boxes, i * 4 + 1))
  );
  const hm = Math.max(medianF32(heights), 1);
  const [ox, oy] = unionBox(boxes);
  return boxes.map((v, k) => f(f(v - (k % 2 === 0 ? ox : oy)) / f(hm)));
}
