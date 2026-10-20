// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
//
// Unicode properties JavaScript lacks, from tables made out of Python's
// unicodedata (Unicode 15.0.0): bidi class, East Asian width, combining marks.

import { BIDI_TABLE, WIDE_TABLE } from "./unicode-data.js";
import { item } from "./util.js";

/** Bidi classes the port tells apart; everything else is `OTHER`. */
export const OTHER = 0;
/** R or AL. */
export const RTL = 1;
/** L. */
export const LTR = 2;
/** EN or AN. */
export const DIGIT = 3;
/** General category Mn or Me. */
export const MARK = 4;

type Ranges = {
  starts: Uint32Array;
  ends: Uint32Array;
  classes: Uint8Array;
};

function decode(table: string): Ranges {
  const t = table.split(" ").map((s) => Number.parseInt(s, 36));
  const n = t.length / 3;
  const starts = new Uint32Array(n);
  const ends = new Uint32Array(n);
  const classes = new Uint8Array(n);
  let end = -1;
  for (let i = 0; i < n; i++) {
    const start = end + 1 + (t[3 * i] ?? 0);
    end = start + (t[3 * i + 1] ?? 0);
    starts[i] = start;
    ends[i] = end;
    classes[i] = t[3 * i + 2] ?? 0;
  }
  return { starts, ends, classes };
}

let bidi: Ranges | undefined;
let wide: Ranges | undefined;

/** The class of range holding `cp`, or 0. */
function lookup(r: Ranges, cp: number): number {
  let lo = 0;
  let hi = r.starts.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cp < item(r.starts, mid)) hi = mid - 1;
    else if (cp > item(r.ends, mid)) lo = mid + 1;
    else return item(r.classes, mid);
  }
  return 0;
}

/** `RTL`, `LTR`, `DIGIT`, `MARK` or `OTHER` for a code point. */
export function bidiClass(cp: number): number {
  bidi ??= decode(BIDI_TABLE);
  return lookup(bidi, cp);
}

/** Whether East Asian width is W or F. */
export function eastAsianWide(cp: number): boolean {
  wide ??= decode(WIDE_TABLE);
  return lookup(wide, cp) === 1;
}

const cpOf = (c: string): number => c.codePointAt(0) ?? 0;

/** Strong right-to-left (bidi R or AL) for the first code point of `c`. */
export function isRtl(c: string): boolean {
  return bidiClass(cpOf(c)) === RTL;
}

/** Reads left to right inside RTL text: bidi L, EN or AN. */
export function isLtr(c: string): boolean {
  const k = bidiClass(cpOf(c));
  return k === LTR || k === DIGIT;
}

/** Bidi class L only (letters, not digits). */
export function isStrongL(c: string): boolean {
  return bidiClass(cpOf(c)) === LTR;
}

/** Combining mark (Mn or Me). */
export function isMark(c: string): boolean {
  return bidiClass(cpOf(c)) === MARK;
}

/** Length in code points with East Asian wide characters counted twice. */
export function displayLen(text: string): number {
  let n = 0;
  for (const c of text) n += eastAsianWide(cpOf(c)) ? 2 : 1;
  return n;
}
