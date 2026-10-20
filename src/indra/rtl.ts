// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/rtl.py (Apache-2.0, Copyright Datalab).
//
// Right-to-left text for a recognizer that reads every line left to right: its
// label is the characters in visual order. `flip` turns logical order into
// visual order and back (it is its own inverse). A small subset of UAX #9: no
// embeddings or isolates, neutrals resolve by runs.

import { isLtr, isMark, isRtl, isStrongL } from "./unicode.js";
import { item } from "./util.js";

export { isRtl };

type Kind = "R" | "L" | "N";

const MIRROR = new Map<string, string>();
{
  const from = String.fromCharCode(0x28, 0x29, 0x5b, 0x5d, 0x7b, 0x7d, 0x3c, 0x3e);
  const to = String.fromCharCode(0x29, 0x28, 0x5d, 0x5b, 0x7d, 0x7b, 0x3e, 0x3c);
  for (let i = 0; i < from.length; i++) MIRROR.set(from.charAt(i), to.charAt(i));
  // Guillemets and single angle quotes: U+00AB U+00BB U+2039 U+203A.
  MIRROR.set(String.fromCharCode(0xab), String.fromCharCode(0xbb));
  MIRROR.set(String.fromCharCode(0xbb), String.fromCharCode(0xab));
  MIRROR.set(String.fromCharCode(0x2039), String.fromCharCode(0x203a));
  MIRROR.set(String.fromCharCode(0x203a), String.fromCharCode(0x2039));
}

// Presentation forms (glyph variants) map to base letters; tatweel carries no letter.
const TATWEEL = 0x640;
const isPresentation = (o: number): boolean =>
  (o >= 0xfb1d && o <= 0xfdff) || (o >= 0xfe70 && o <= 0xfeff);

/** Whether any character is strong right to left. */
export function hasRtl(text: string): boolean {
  for (const c of text) if (isRtl(c)) return true;
  return false;
}

/**
 * Whether the line reads right to left: more RTL letters than LTR letters. The
 * counts do not depend on the order, so logical and visual text agree.
 */
export function rtlBase(text: string): boolean {
  let r = 0;
  let l = 0;
  for (const c of text) {
    if (isRtl(c)) r++;
    else if (isStrongL(c)) l++;
  }
  return r > l;
}

/**
 * Presentation forms to base letters (NFKC on those blocks only), no tatweel,
 * no combining mark without a base character.
 */
export function normalize(text: string): string {
  const out: string[] = [];
  for (const c of text) {
    const o = c.codePointAt(0) ?? 0;
    if (isPresentation(o)) out.push(c.normalize("NFKC"));
    else if (o !== TATWEEL && (out.length > 0 || !isMark(c))) out.push(c);
  }
  return out.join("");
}

/** Base characters with their combining marks (harakat, niqqud). */
export function clusters(text: string): string[] {
  const out: string[] = [];
  for (const c of text) {
    if (out.length > 0 && isMark(c)) out[out.length - 1] += c;
    else out.push(c);
  }
  return out;
}

function kindOf(cl: string): Kind {
  if (isRtl(cl)) return "R";
  return isLtr(cl) ? "L" : "N";
}

/** [start, end) spans from a `strong` cluster to the last one before a `stop` cluster. */
function runs(kinds: Kind[], strong: Kind, stop: Kind): [number, number][] {
  const spans: [number, number][] = [];
  let start = -1;
  let last = -1;
  for (let i = 0; i <= kinds.length; i++) {
    const k = kinds[i];
    if (k === strong) {
      if (start < 0) start = i;
      last = i;
    } else if (k === stop || k === undefined) {
      if (start >= 0) spans.push([start, last + 1]);
      start = last = -1;
    }
  }
  return spans;
}

function mirrored(cl: string): string {
  const first = String.fromCodePoint(cl.codePointAt(0) ?? 0);
  return (MIRROR.get(first) ?? first) + cl.slice(first.length);
}

/**
 * Logical order to visual order, and visual order back to logical order.
 * `baseRtl` is the paragraph direction when known; null or undefined decides by `rtlBase`.
 */
export function flip(text: string, mirror = true, baseRtl?: boolean | null): string {
  if (!hasRtl(text)) return text;
  const cls = clusters(text);
  const kinds = cls.map(kindOf);
  if (baseRtl ?? rtlBase(text)) {
    // Units: LTR runs kept whole, every other cluster alone; reverse the units.
    const units: string[] = [];
    let i = 0;
    const spans = runs(kinds, "L", "R");
    spans.push([cls.length, cls.length]);
    for (const [a, b] of spans) {
      for (let j = i; j < a; j++) units.push(mirror ? mirrored(item(cls, j)) : item(cls, j));
      if (a < b) units.push(cls.slice(a, b).join(""));
      i = b;
    }
    return units.reverse().join("");
  }
  // LTR base: reverse each RTL run in place.
  const out = [...cls];
  for (const [a, b] of runs(kinds, "R", "L")) {
    const seg = cls.slice(a, b).reverse();
    out.splice(a, b - a, ...(mirror ? seg.map(mirrored) : seg));
  }
  return out.join("");
}

/** The recognizer's label for a line stored in logical order. */
export function visualLabel(logical: string, baseRtl?: boolean | null): string {
  return flip(normalize(logical), true, baseRtl);
}

/**
 * The label for characters already in visual order, as stored (a PDF text layer
 * read by glyph position). `shapes`: the layer stores each bracket as the shape
 * drawn (see `bracketsAsShapes`), so it is already mirrored.
 */
export function labelFromVisual(rawVisual: string, shapes = false): string {
  return visualLabel(flip(rawVisual, shapes));
}

/**
 * Whether a page's text layer stores RTL brackets as drawn shapes. A page uses
 * one convention, so its RTL lines without Latin letters vote by their first bracket.
 */
export function bracketsAsShapes(rawVisualLines: string[]): boolean {
  let shapes = 0;
  let logical = 0;
  for (const t of rawVisualLines) {
    if (!rtlBase(t) || [...t].some(isStrongL)) continue;
    for (const c of t) {
      if ("([{".includes(c)) {
        shapes++;
        break;
      }
      if (")]}".includes(c)) {
        logical++;
        break;
      }
    }
  }
  return shapes > logical;
}
