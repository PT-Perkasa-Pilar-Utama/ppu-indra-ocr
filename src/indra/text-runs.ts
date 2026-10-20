// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/textlayer.py (Apache-2.0, Copyright Datalab).

import { pyRound } from "./det-post.js";
import type { Box } from "./rec-preproc.js";
import { isRtl } from "./rtl.js";
import type { ExtractedRuns, PdfPageSource, TextLayerStats, TextLine } from "./textlayer.js";
import { isPySpace, item } from "./util.js";

/** Neighbouring right-to-left letters inside runs, and the share that step left to right. */
export type RtlPairs = { pairs: number; share: number };

const BAD = /^[\p{Co}\p{Cs}\p{Cn}]$/u;
const CONTROL = /^\p{Cc}$/u;

/** Replacement, private-use, surrogate, unassigned and control characters (tab excepted). */
export function badCharCount(text: string): number {
  let n = 0;
  // By code point, as Python: a lone surrogate counts as bad (category Cs).
  for (const c of text) {
    if (c.codePointAt(0) === 0xfffd || BAD.test(c) || (CONTROL.test(c) && c !== "\t")) n++;
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

/**
 * Quarter turns (counter-clockwise, as np.rot90) that a baseline running in
 * direction (dx, dy) (rendered frame, y down) is turned from left to right;
 * undefined when it is not a quarter turn.
 */
function quarter(dx: number, dy: number): number | undefined {
  const q = Math.atan2(dy, dx) / (Math.PI / 2);
  const k = pyRound(q);
  return Math.abs(q - k) < 0.01 ? ((k % 4) + 4) % 4 : undefined;
}

/** Rendered-frame points in the frame of the page turned upright again (np.rot90(gray, -turn)). */
function turnFrame(x: number, y: number, w: number, h: number, turn: number): [number, number] {
  if (turn === 1) return [h - y, x];
  if (turn === 2) return [w - x, h - y];
  if (turn === 3) return [y, w - x];
  return [x, y];
}

/**
 * Whether a page's upright characters run top to bottom (vertical writing):
 * most steps between neighbouring characters go down, not right.
 */
export function verticalLayout(stats: TextLayerStats): boolean {
  const [down, right] = stats.steps ?? [0, 0];
  return down >= 30 && down > 2 * right;
}

/** The quarter turn that holds most of the page's characters (a table set sideways), or 0. */
export function pageTurn(stats: TextLayerStats): number {
  const q = stats.quarters;
  let k = 1;
  for (const i of [2, 3]) if (q[i] > q[k]) k = i;
  return q[k] > 0.5 * Math.max(stats.chars, 1) ? k : 0;
}

/**
 * The character of a UTF-16 unit `u` (`u2`: the next unit), and whether it
 * used `u2`. A character above U+FFFF comes as two surrogates; a lone
 * surrogate becomes U+FFFD.
 */
export function codePoint(u: number, u2 = 0): [string, boolean] {
  if (u >= 0xd800 && u <= 0xdbff && u2 >= 0xdc00 && u2 <= 0xdfff) {
    return [String.fromCodePoint(0x10000 + ((u - 0xd800) << 10) + (u2 - 0xdc00)), true];
  }
  if (u >= 0xd800 && u <= 0xdfff) return [String.fromCharCode(0xfffd), false];
  return [String.fromCharCode(u), false];
}

/**
 * Text runs of a page that read left to right once the page is turned upright
 * by `turn` quarter turns (see pageTurn), in content order, and character
 * counts. Boxes are in the frame turned by `frame` (default `turn`): a
 * vertical page has upright characters (turn 0) read in a turned frame (3).
 */
export function extractRuns(page: PdfPageSource, turn = 0, frame: number = turn): ExtractedRuns {
  const [a00 = 0, a01 = 0, a02 = 0, a10 = 0, a11 = 0, a12 = 0] = deviceAffine(page);
  // pdfium's char angle turns clockwise with y up, and quarter() counts the
  // other way with y down: F L F (F flips y) applies both.
  const l00 = a00;
  const l01 = -a01;
  const l10 = -a10;
  const l11 = a11;
  const quarters = new Map<number, number | undefined>();
  const chars: string[] = [];
  const raw: (readonly [number, number, number, number])[] = [];
  const breaks: number[] = []; // runs end after these positions
  const qs: number[] = []; // each character's quarter turn relative to `turn`
  let curQ = 0;
  const stats: TextLayerStats = { chars: 0, rotated: 0, bad: 0, quarters: [0, 0, 0, 0] };
  const all = page.chars();
  let low = false; // this index is the second half of a surrogate pair
  for (const [i, ch] of all.entries()) {
    if (low) {
      low = false;
      continue;
    }
    const u2 = ch.unit >= 0xd800 && ch.unit <= 0xdbff ? (all[i + 1]?.unit ?? 0) : 0;
    let c: string;
    [c, low] = codePoint(ch.unit, u2);
    if (c === "\r" || c === "\n") {
      breaks.push(chars.length);
      continue;
    }
    if (c === "\u0002") c = "-"; // pdfium's mark for a hyphen at a line break
    if (!isPySpace(c)) {
      stats.chars++;
      stats.bad += badCharCount(c);
      let q = quarters.get(ch.angle);
      if (!quarters.has(ch.angle)) {
        const cos = Math.cos(ch.angle);
        const sin = Math.sin(ch.angle);
        q = quarter(l00 * cos + l01 * sin, l10 * cos + l11 * sin);
        quarters.set(ch.angle, q);
      }
      if (q !== undefined) {
        // SAFETY: quarter() returns 0..3 when it returns a number.
        stats.quarters[q as 0 | 1 | 2 | 3]++;
      }
      if (q !== turn) stats.rotated++; // not upright on the rendered page
      if (q === undefined) {
        // not a quarter turn: not read
        breaks.push(chars.length);
        continue;
      }
      curQ = (((q - turn) % 4) + 4) % 4;
    }
    if (!ch.box) throw new Error("Failed to get charbox.");
    chars.push(c);
    qs.push(curQ);
    raw.push(ch.box);
  }
  if (!chars.length) return { runs: [], stats };
  const [w, h] = page.size();
  const corners = raw.map(([l, b, r, t]) => {
    const xs = [l, r, l, r];
    const ys = [b, b, t, t];
    return {
      px: xs.map((x, k) => a00 * x + a01 * item(ys, k) + a02),
      py: xs.map((x, k) => a10 * x + a11 * item(ys, k) + a12),
    };
  });
  let boxes = corners.map(({ px, py }): Box => [
    Math.min(...px),
    Math.min(...py),
    Math.max(...px),
    Math.max(...py),
  ]);
  // Characters outside the rendered frame (beyond a crop box) are not on the page.
  const cx = boxes.map(([x0, , x1]) => (x0 + x1) / 2);
  const cy = boxes.map(([, y0, , y1]) => (y0 + y1) / 2);
  const visible = boxes.map((_, i) => {
    const x = item(cx, i);
    const y = item(cy, i);
    return x >= 0 && x <= w && y >= 0 && y <= h;
  });
  stats.off_page = visible.filter((v) => !v).length;
  // Steps between neighbouring upright characters of a run: down (vertical
  // writing) or right (horizontal), in the rendered frame. A jump to another
  // run is not a step.
  const cut = new Set(breaks.filter((b) => b > 0 && b < chars.length).map((b) => b - 1));
  let down = 0;
  let right = 0;
  for (let i = 0; i + 1 < chars.length; i++) {
    if (cut.has(i) || item(qs, i) || item(qs, i + 1)) continue;
    if (isPySpace(item(chars, i)) || isPySpace(item(chars, i + 1))) continue;
    const b = item(boxes, i);
    const ch = Math.max(b[3] - b[1], 1e-3);
    const dx = Math.abs(item(cx, i + 1) - item(cx, i));
    const dyRaw = item(cy, i + 1) - item(cy, i);
    const dy = Math.abs(dyRaw);
    if (dx > 3 * ch || dy > 3 * ch) continue;
    if (dyRaw > 0.3 * ch && dx < 0.3 * ch) down++;
    // Horizontal steps go right (left-to-right scripts) or left (right to left).
    if (dx > 0.3 * ch && dy < 0.3 * ch) right++;
  }
  stats.steps = [down, right];
  if (frame) {
    boxes = corners.map(({ px, py }): Box => {
      const pts = px.map((x, k) => turnFrame(x, item(py, k), w, h, frame));
      const tx = pts.map((p) => p[0]);
      const ty = pts.map((p) => p[1]);
      return [Math.min(...tx), Math.min(...ty), Math.max(...tx), Math.max(...ty)];
    });
  }
  // A run also ends where the quarter turn changes (a sideways label in an
  // upright line of characters).
  const turns = qs.flatMap((q, i) => (i > 0 && q !== qs[i - 1] ? [i] : []));
  const runs: TextLine[] = [];
  let start = 0;
  for (const end of [...new Set([...breaks, ...turns, chars.length])].toSorted((p, q) => p - q)) {
    const seg: [string, Box][] = [];
    for (let i = start; i < end; i++) if (visible[i]) seg.push([item(chars, i), item(boxes, i)]);
    let q = start < qs.length ? item(qs, start) : 0;
    start = end;
    const ink = seg.filter(([c]) => !isPySpace(c)).map(([, b]) => b);
    if (!ink.length) continue;
    if (frame !== turn && !q && ink.length > 1) {
      // A vertical page read turned: a horizontal run (a header, a caption)
      // runs top to bottom here, so it is a sideways line.
      const first = item(ink, 0);
      const last = item(ink, ink.length - 1);
      const dx = Math.abs((last[0] + last[2]) / 2 - (first[0] + first[2]) / 2);
      const dy = Math.abs((last[1] + last[3]) / 2 - (first[1] + first[3]) / 2);
      if (dy > 2 * dx) q = 1;
    }
    runs.push({
      bbox: [
        Math.min(...ink.map((b) => b[0])),
        Math.min(...ink.map((b) => b[1])),
        Math.max(...ink.map((b) => b[2])),
        Math.max(...ink.map((b) => b[3])),
      ],
      text: collapseSpaces(seg.map(([c]) => c).join("")),
      chars: seg,
      quarter: q,
      seq: runs.length,
    });
  }
  return { runs, stats };
}

/**
 * Neighbouring right-to-left letters inside a run, and the share of them that
 * step left to right on the page.
 */
export function visualRtlFrac(runs: readonly TextLine[]): RtlPairs {
  let pairs = 0;
  let right = 0;
  for (const r of runs) {
    for (let i = 1; i < r.chars.length; i++) {
      const [a, ba] = item(r.chars, i - 1);
      const [b, bb] = item(r.chars, i);
      if (isRtl(a) && isRtl(b)) {
        pairs++;
        if (bb[0] + bb[2] > ba[0] + ba[2]) right++;
      }
    }
  }
  return { pairs, share: right / Math.max(pairs, 1) };
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
