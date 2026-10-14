// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { Box } from "./rec-preproc.js";
import type { LinePart, RecLine, Tok } from "./types.js";
import { item } from "./util.js";
import type { Vocab } from "./vocab.js";

// Gaps between CTC character positions (centers), in line heights. The label
// rules (2.5 and 1.0 line heights) measure glyph edges; centers add about half
// a line height of glyph width.
export const SPLIT_GAP = 3.0; // cut a detected line at a wider gap (outside tables)
export const CELL_GAP = 1.8; // cut a table line into cells at a wider gap
const GUTTER_GAP = 1.6; // cut at a smaller gap when other lines' edges line up with it
const GUTTER_EDGES = 3; // how many other line edges must line up (within half a line height)

/**
 * Per line, the x positions of gaps that sit on a column gutter: a gap over
 * GUTTER_GAP line heights where at least GUTTER_EDGES other lines end at its
 * left side or start at its right side.
 */
export function gutterCuts(
  vocab: Vocab,
  boxes: readonly Box[],
  rec: readonly RecLine[]
): number[][] {
  const f = Math.fround;
  return boxes.map((b, i) => {
    const h = Math.max(1, f(b[3] - b[1]));
    const xs = item(rec, i)
      .toks.filter((tk) => !isSpace(vocab, tk[0]))
      .map((tk) => tk[1]);
    const mine: number[] = [];
    for (let k = 1; k < xs.length; k++) {
      const a = item(xs, k - 1);
      const c = item(xs, k);
      if (c - a <= GUTTER_GAP * h) continue;
      const gl = f(a + 0.35 * h); // gap edges, compared in float32 as numpy does
      const gr = f(c - 0.35 * h);
      const tol = f(0.5 * h);
      let near = 0;
      for (const [j, o] of boxes.entries()) {
        if (j !== i && (Math.abs(f(o[2] - gl)) < tol || Math.abs(f(o[0] - gr)) < tol)) near++;
      }
      if (near >= GUTTER_EDGES) mine.push((a + c) / 2);
    }
    return mine;
  });
}

export function isSpace(vocab: Vocab, tok: number | string): boolean {
  return tok === " " || tok === vocab.index.get(" ");
}

/** Text of tokens: class ids from the recognizer, or characters (CJK, text layer, math). */
export function decodeToks(vocab: Vocab, toks: readonly Tok[]): string {
  if (!toks.length) return vocab.decode([]);
  if (typeof item(toks, 0)[0] === "string") {
    return toks
      .map((tk) => String(tk[0]))
      .join("")
      .split(/\s+/u)
      .filter(Boolean)
      .join(" ");
  }
  return vocab.decode(toks.map((tk) => Number(tk[0])));
}

/**
 * Cut a line where two neighbouring characters are more than `gap` line
 * heights apart, or at the given x positions. One part when nothing is cut.
 */
export function splitLine(
  vocab: Vocab,
  box: Box,
  toks: readonly Tok[],
  gap: number,
  at: readonly number[] = []
): LinePart[] {
  const h = Math.max(1, Math.fround(box[3] - box[1]));
  const cutsAt = at.toSorted((p, q) => p - q);
  const segs: Tok[][] = [[]];
  let prev: number | undefined;
  for (const tk of toks) {
    const x = tk[1];
    if (!isSpace(vocab, tk[0])) {
      if (
        prev !== undefined &&
        (x - prev > gap * h || cutsAt.some((c) => prev !== undefined && prev < c && c < x))
      ) {
        segs.push([]);
      }
      prev = x;
    }
    segs.at(-1)?.push(tk);
  }
  if (segs.length === 1) return [{ box, text: decodeToks(vocab, toks), toks }];
  const out: LinePart[] = [];
  for (const sg of segs) {
    const text = decodeToks(vocab, sg).trim();
    const xs = sg.filter((tk) => !isSpace(vocab, tk[0])).map((tk) => tk[1]);
    if (text && xs.length) {
      out.push({
        box: [
          Math.max(box[0], Math.min(...xs) - 0.35 * h),
          box[1],
          Math.min(box[2], Math.max(...xs) + 0.35 * h),
          box[3],
        ],
        text,
        toks: sg,
      });
    }
  }
  return out;
}
