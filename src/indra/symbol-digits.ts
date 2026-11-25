// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/document.py (Apache-2.0, Copyright Datalab).

import type { Gray } from "./image.js";
import type { IndraPipeline } from "./pipeline.js";
import type { Box } from "./rec-preproc.js";
import { SequenceMatcher } from "./sequence-matcher.js";
import type { Tok } from "./types.js";
import { item } from "./util.js";

/**
 * A digit that may be a symbol glyph (a font that gives a symbol a digit's
 * code, as a minus read '2') is checked against the page: its line is read, a
 * digit that the reading has as a symbol is a candidate, and the symbol
 * replaces the digit when the recognizer finds the line with the symbol at
 * least SYMBOL_MARGIN nats more likely than with the digit.
 */
const SYMBOL_MARGIN = 2.0;
/** A symbol or a dash: what a font that codes glyphs as digits holds (Unicode Sm, So, Sk, Sc, Pd). */
const SYMBOL = /^[\p{Sm}\p{So}\p{Sk}\p{Sc}\p{Pd}]$/u;
const DIGIT = /^[0-9]$/u;

const f = Math.fround;

/** A word of a line holding digits inside symbol-glyph boxes: tokens [start, end) and the digits' positions. */
export type SymbolWord = { start: number; end: number; hits: number[] };

/** The symbol boxes that share a text row with a line spanning y0..y1 (more than half the smaller height). */
function rowBoxes(sym: readonly Box[], y0: number, y1: number): Box[] {
  return sym.filter((s) => {
    const over = f(Math.min(s[3], f(y1)) - Math.max(s[1], f(y0)));
    return over > f(0.5 * Math.min(f(s[3] - s[1]), f(f(y1) - f(y0))));
  });
}

/**
 * Words of one detected line (tokens (char, x, ...), box in pixels) that hold
 * a digit inside a symbol-glyph box (`sym`, pixels).
 */
export function symbolWords(toks: readonly Tok[], box: Box, sym: readonly Box[]): SymbolWord[] {
  if (!sym.length || !toks.length) return [];
  const rows = rowBoxes(sym, box[1], box[3]);
  if (!rows.length) return [];
  const out: SymbolWord[] = [];
  let start: number | undefined;
  for (let j = 0; j <= toks.length; j++) {
    const space = j === toks.length || item(toks, j)[0] === " ";
    if (!space && start === undefined) start = j;
    if (space && start !== undefined) {
      const hits: number[] = [];
      for (let p = start; p < j; p++) {
        const [c, x] = item(toks, p);
        // numpy compares the float32 boxes with the x rounded to float32
        if (
          typeof c === "string" &&
          DIGIT.test(c) &&
          rows.some((r) => f(r[0] - 1) <= f(x) && f(x) <= f(r[2] + 1))
        ) {
          hits.push(p);
        }
      }
      if (hits.length) out.push({ start, end: j, hits });
      start = undefined;
    }
  }
  return out;
}

/**
 * Positions (into `layer`, a line's text from its tokens) of the digits at
 * `hits` that the recognizer's reading `read` has as a symbol or a dash, one
 * character for one, and that character. A cut or tiny glyph reads as light
 * punctuation ('.', ':', '_'), which does not count.
 */
export function symbolCandidates(
  layer: string,
  read: string,
  hits: readonly number[]
): Map<number, string> {
  const out = new Map<number, string>();
  const hit = new Set(hits);
  const a = Array.from(layer);
  const b = Array.from(read);
  for (const [op, i1, i2, j1, j2] of new SequenceMatcher(a, b).getOpcodes()) {
    if (op !== "replace" || i2 - i1 !== j2 - j1) continue;
    for (let p = i1; p < i2; p++) {
      const c = item(b, j1 + p - i1);
      if (hit.has(p) && SYMBOL.test(c)) out.set(p, c);
    }
  }
  return out;
}

/**
 * Check the layer digits that lie in a possible symbol glyph's box (`sym`,
 * pixels) against the page, and put the symbol in place of a digit where the
 * recognizer supports it; `toks` changes in place. Returns the number of
 * digits replaced.
 */
export async function fixSymbolDigits(
  pipeline: IndraPipeline,
  gray: Gray,
  boxes: readonly Box[],
  toks: Tok[][],
  sym: readonly Box[],
  skip: ReadonlySet<number>
): Promise<number> {
  const lines = new Map<number, number[]>();
  for (const [i, tk] of toks.entries()) {
    if (!tk.length || skip.has(i)) continue;
    const hits = symbolWords(tk, item(boxes, i), sym).flatMap((w) => w.hits);
    if (hits.length) lines.set(i, hits);
  }
  if (!lines.size) return 0;
  const idx = [...lines.keys()].toSorted((p, q) => p - q);
  // Each line is read in a box that holds the whole of every checked glyph: a
  // detected box can stop inside a line's last glyph, and a cut digit reads as
  // a symbol ('2' as '.').
  const rbox = boxes.map((b): [number, number, number, number] => [
    f(b[0]),
    f(b[1]),
    f(b[2]),
    f(b[3]),
  ]);
  for (const i of idx) {
    const b = item(rbox, i);
    const [, y0, , y1] = b;
    const xs = (lines.get(i) ?? []).map((p) => f(item(item(toks, i), p)[1]));
    const near = rowBoxes(sym, y0, y1).filter((s) =>
      xs.some((x) => f(s[0] - 1) <= x && x <= f(s[2] + 1))
    );
    if (near.length) {
      const pad = 0.3 * (y1 - y0);
      b[0] = f(Math.min(b[0], Math.min(...near.map((s) => s[0])) - pad));
      b[2] = f(Math.max(b[2], Math.max(...near.map((s) => s[2])) + pad));
    }
  }
  for (const b of rbox) {
    b[0] = Math.min(Math.max(b[0], 0), gray.width - 1);
    b[2] = Math.min(Math.max(b[2], 0), gray.width - 1);
  }
  const reads = await pipeline.recognize(
    gray,
    idx.map((i) => item(rbox, i))
  );
  const candBoxes: Box[] = [];
  const candToks: Tok[][] = [];
  const owners: { i: number; p: number; c: string; length: number }[] = [];
  for (const [n, i] of idx.entries()) {
    const tk = item(toks, i);
    const layer = tk.map((t) => (typeof t[0] === "string" ? t[0] : " ")).join("");
    for (const [p, c] of symbolCandidates(layer, item(reads, n).text, lines.get(i) ?? [])) {
      const alt = [...tk];
      const [, x, ...rest] = item(alt, p);
      alt[p] = [c, x, ...rest];
      candBoxes.push(item(rbox, i), item(rbox, i));
      candToks.push(tk, alt);
      owners.push({ i, p, c, length: layer.replaceAll(" ", "").length });
    }
  }
  if (!owners.length) return 0;
  const gaps = await pipeline.layerGaps(gray, candBoxes, candToks);
  let fixed = 0;
  for (const [k, { i, p, c, length }] of owners.entries()) {
    const gLayer = item(gaps, 2 * k).gap;
    const gAlt = item(gaps, 2 * k + 1).gap;
    if (gLayer === null || gAlt === null) continue;
    if ((gLayer - gAlt) * Math.max(length, 1) >= SYMBOL_MARGIN) {
      const tk = item(toks, i);
      const [, x, ...rest] = item(tk, p);
      tk[p] = [c, x, ...rest];
      fixed++;
    }
  }
  return fixed;
}
