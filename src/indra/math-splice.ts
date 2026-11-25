// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/math.py (Apache-2.0, Copyright Datalab).
//
// Prose splice: a MathRec reading holds Latin letters without accents, so the prose
// around its formulas is aligned with the plain reading (the recognizer's or the text
// layer's) on a folded form, and the plain reading's letters go back in.

import {
  greekSmallLetter,
  isAlpha,
  isCombining,
  pyLstrip,
  pyRstrip,
  pyStrip,
} from "./py-unicode.js";
import { SequenceMatcher } from "./sequence-matcher.js";
import { isPySpace, item } from "./util.js";

/** A prose segment is spliced only when this share of its characters align with the plain reading. */
export const SPLICE_MIN_SEG = 0.5;
/** ... and only when the line's prose as a whole reaches this share: else the line keeps today's reading. */
export const SPLICE_MIN_LINE = 0.6;

// Python's \s (str.isspace) as the inside of a character class: JS \s has U+FEFF and
// lacks U+001C..U+001F and U+0085.
export const SP =
  "\\t-\\r\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
// Python's \b after a word character: the next character is not a letter, digit or "_".
const WORD_END = "(?![\\p{L}\\p{N}_])";
export const TEXT_STYLE =
  "mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|boldsymbol|textbf|textit|textrm|text|operatorname";

// Commands that style or space text, and escaped symbols (\%): no math of their own.
export const STYLE_SOURCE: string = `\\\\(?:${TEXT_STYLE})${WORD_END}|\\\\[,;:! %#&$]|\\\\q?quad${WORD_END}|[{}~${SP}]`;
const STYLE_FULL = new RegExp(`^(?:${STYLE_SOURCE})$`, "u");

// An accent written as math over one letter: MathRec has no accented letters, so
// "e with acute" may come out as \hat { \epsilon } and "z with dot" as \bar { z }.
export const ACCENT: RegExp = new RegExp(
  `^[${SP}]*\\\\(?:hat|bar|dot|ddot|tilde|acute|grave|breve|check)[${SP}]*\\{[${SP}]*\\\\?[A-Za-z]{1,8}[${SP}]*\\}[${SP}]*$`,
  "u"
);
const ASCII_LETTER = /^[A-Za-z]$/;
// a $...$ span inside a prose segment is an accent span; split keeps it as its own piece
const ACCENT_SPAN = /(\$[^$]*\$)/;
const GREEK = /^\\(?:var)?([a-z]+)$/;

// Letters that look alike across Latin, Cyrillic and Greek, and Latin letters without
// a decomposition, map to one Latin letter (see look).
const LOOK_FROM =
  "430 432 441 435 4bb 456 458 43a 43c 43d 43e 440 455 442 443 445 44c 501 51b 51d 433 454 3b1 3b2 3b5 3b7 3b9 3ba 3bd 3bf 3c1 3c4 3c5 3c7 3c9 3f2 3f3 131 237 142 111 f0 127 167 f8 153 e6 df fe 1bf 259 25b 254";
const LOOK_TO = "abcehijkmhopstyxbdqwreabenikvoptuxwcjijlddhtooabppeeo";
const LOOK = new Map(
  LOOK_FROM.split(" ").map((h, i): [number, string] => [Number.parseInt(h, 16), LOOK_TO.charAt(i)])
);

/** A character without its accents, lower case (one character out): `_fold`. */
export function fold(c: string): string {
  const base = Array.from(c.normalize("NFKD")).filter((x) => !isCombining(x));
  return (base[0] ?? c).toLowerCase();
}

/** `_fold(c) != c.lower()`: the character has an accent. */
export const accentedLetter = (c: string): boolean => isAlpha(c) && fold(c) !== c.toLowerCase();
export const hasAccents = (draft: string): boolean => Array.from(draft).some(accentedLetter);

/**
 * MathRec's reading with the accents of the plain reading put back in its prose:
 * MathRec writes Latin letters without accents ("duvodu" for "d u-with-hacek vodu").
 * Only prose letters that match a letter of the plain reading change; the $...$ spans
 * stay as MathRec wrote them.
 */
export function restoreAccents(reading: string, draft: string): string {
  if (!pyStrip(draft)) return reading;
  const chars = reading.split("$").map((p) => Array.from(p));
  // (part index, char index) of every prose letter
  const pos: [number, number][] = [];
  for (let k = 0; k < chars.length; k += 2) {
    item(chars, k).forEach((c, i) => {
      if (isAlpha(c)) pos.push([k, i]);
    });
  }
  const b = Array.from(draft).filter(isAlpha);
  if (!pos.length || !b.length) return reading;
  const sm = new SequenceMatcher(
    pos.map(([k, i]) => fold(item(item(chars, k), i))),
    b.map(fold)
  );
  for (const [op, a0, a1, b0, b1] of sm.getOpcodes()) {
    // letters that match, and short one-for-one swaps (MathRec read a hacek u as "i")
    if (op !== "equal" && !(op === "replace" && a1 - a0 === b1 - b0 && b1 - b0 <= 2)) continue;
    for (let t = 0; t < a1 - a0; t++) {
      const [k, i] = item(pos, a0 + t);
      const c = item(b, b0 + t);
      if (accentedLetter(c)) item(chars, k)[i] = c;
    }
  }
  return chars.map((p) => p.join("")).join("$");
}

/** One character for the alignment: no accents, lower case, look-alike letters as one Latin letter. */
export const look = (c: string): string =>
  Array.from(fold(c), (x) => LOOK.get(x.codePointAt(0) ?? 0) ?? x).join("");
/** An alignment unit: (folded unit, start, end) in code points of the text. */
export type Unit = readonly [unit: string, start: number, end: number];

const isAsciiLetter = (c: string | undefined): boolean => c !== undefined && ASCII_LETTER.test(c);

/**
 * Alignment units of a text, spaces left out. A LaTeX command is one unit (a Greek
 * letter as its look-alike), and the structure of a $...$ span (braces, ^, _, styling
 * commands) writes none, so a span compares with the plain characters it stands for.
 * A plain reading may hold commands as well ("Burada \sigma ana").
 */
export function units(text: string): Unit[] {
  const cps = Array.from(text);
  const out: Unit[] = [];
  for (let i = 0; i < cps.length;) {
    const c = item(cps, i);
    let end = i + 1;
    if (c === "\\") while (isAsciiLetter(cps[end])) end++;
    const tok = cps.slice(i, end).join("");
    const start = i;
    i = end;
    if (isPySpace(c)) continue;
    if (tok.startsWith("\\")) {
      if (STYLE_FULL.test(tok)) continue;
      const g = GREEK.exec(tok);
      const letter = g?.[1] === undefined ? undefined : greekSmallLetter(g[1]);
      out.push([letter === undefined ? tok : look(letter), start, end]);
    } else if (!"{}^_~".includes(tok)) {
      out.push([look(tok), start, end]);
    }
  }
  return out;
}

/** Counts of the non-ASCII letters of a text (Python's Counter in `_letters`). */
function letters(s: string): Map<string, number> {
  const n = new Map<string, number>();
  for (const c of s) {
    if (isAlpha(c) && (c.codePointAt(0) ?? 0) > 127) n.set(c, (n.get(c) ?? 0) + 1);
  }
  return n;
}

/** Counter subtraction is not empty: `sub` has a letter that `t` has not, or more of it. */
function hasMoreLetters(sub: string, t: string): boolean {
  const have = letters(t);
  for (const [c, count] of letters(sub)) if (count > (have.get(c) ?? 0)) return true;
  return false;
}

/** A segment of a reading: prose (outside the $...$ spans) or a formula span. */
type Segment = { prose: boolean; reading: string; today: string };

/** Splits the reading into segments; an accent span joins the prose around it. */
function segments(
  parts: readonly string[],
  tparts: readonly string[],
  accented: boolean
): Segment[] {
  const segs: Segment[] = [];
  parts.forEach((part, k) => {
    let p = part;
    let t = item(tparts, k);
    if (k % 2 === 1 && !(accented && ACCENT.test(p))) {
      segs.push({ prose: false, reading: p, today: t });
      return;
    }
    if (k % 2 === 1) {
      p = `$${p}$`;
      t = `$${t}$`;
    }
    const last = segs.at(-1);
    if (last?.prose) {
      last.reading += p;
      last.today += t;
    } else {
      segs.push({ prose: true, reading: p, today: t });
    }
  });
  return segs;
}

/** Alignment units of the reading: (folded unit, segment index). */
function readingUnits(segs: readonly Segment[]): [string, number][] {
  const out: [string, number][] = [];
  segs.forEach((seg, n) => {
    if (!seg.prose) {
      for (const [u] of units(seg.reading)) out.push([u, n]);
      return;
    }
    for (const piece of seg.reading.split(ACCENT_SPAN)) {
      if (piece.length > 1 && piece.startsWith("$") && piece.endsWith("$")) {
        out.push(["\0", n]); // an accent span: one letter of the plain reading
      } else {
        for (const c of piece) if (!isPySpace(c)) out.push([look(c), n]);
      }
    }
  });
  return out;
}

/** Per reading unit: the plain units [start, end) it stands for, and whether they are equal. */
type Alignment = { start: number[]; end: number[]; hit: boolean[] };

function align(kindOf: readonly boolean[], u: readonly string[], d: readonly string[]): Alignment {
  const start: number[] = Array.from({ length: u.length }, () => 0);
  const end: number[] = Array.from({ length: u.length }, () => 0);
  const hit: boolean[] = Array.from({ length: u.length }, () => false);
  for (const [op, a0, a1, b0, b1] of new SequenceMatcher(u, d).getOpcodes()) {
    if (op === "equal") {
      for (let i = a0; i < a1; i++) {
        start[i] = b0 + i - a0;
        end[i] = b0 + i - a0 + 1;
        hit[i] = true;
      }
      continue;
    }
    const kinds = kindOf.slice(a0, a1);
    const first = kinds[0] === true;
    let cut = -1;
    for (let i = 1; i < kinds.length && cut < 0; i++) if (kinds[i] !== kinds[0]) cut = i;
    let mid: number | null = null;
    if (cut >= 0 && kinds.slice(cut).every((k) => k === kinds[cut])) {
      // one prose run and one formula run: a misread letter is one letter (o for
      // schwa), so the prose run takes one plain unit per unit at its edge and the
      // formula the rest
      const take = Math.min(first ? cut : kinds.length - cut, b1 - b0);
      mid = first ? b0 + take : b1 - take;
    }
    for (let i = a0; i < a1; i++) {
      const r = i - a0;
      if (mid === null) {
        // spread the plain units over the units
        start[i] = b0 + Math.floor((r * (b1 - b0)) / (a1 - a0));
        end[i] = b0 + Math.floor(((r + 1) * (b1 - b0)) / (a1 - a0));
      } else if (first) {
        [start[i], end[i]] =
          r < cut ? [Math.min(b0 + r, mid), Math.min(b0 + r + 1, mid)] : [mid, b1];
      } else {
        const q = r - cut;
        [start[i], end[i]] =
          r < cut ? [b0, mid] : [Math.min(mid + q, b1), Math.min(mid + q + 1, b1)];
      }
    }
  }
  return { start, end, hit };
}

const count = (flags: readonly boolean[], ids: readonly number[]): number =>
  ids.reduce((n, i) => n + (flags[i] ? 1 : 0), 0);

/**
 * MathRec's reading with its prose (the text outside $...$) taken from the plain
 * reading: MathRec writes LaTeX and Latin letters, so prose around a formula loses
 * accents and letters of other scripts. The reading's prose characters and its spans'
 * plain characters are aligned with the plain reading on a folded form (look). Each
 * prose segment maps to the plain characters between its first and last aligned
 * position, and that substring replaces the segment when it holds non-ASCII letters
 * that the segment lacks. A span that only writes an accent over one letter counts as
 * prose when the plain reading has accents.
 *
 * today: the text without the splice (default: restoreAccents). It comes back
 * unchanged when nothing is lost, when the line is all math, or when the alignment is
 * not sure (SPLICE_MIN_SEG, SPLICE_MIN_LINE).
 */
export function spliceProse(
  reading: string,
  draft: string,
  today: string = restoreAccents(reading, draft)
): string {
  if (!pyStrip(draft) || letters(draft).size === 0 || !reading.includes("$")) return today;
  const parts = reading.split("$");
  const tparts = today.split("$");
  if (parts.length !== tparts.length || parts.length % 2 === 0) return today; // no clean segments
  const segs = segments(parts, tparts, hasAccents(draft));
  if (!segs.some((s) => s.prose && Array.from(s.reading).some(isAlpha))) return today; // all math
  const ru = readingUnits(segs);
  const du = units(draft);
  const { start, end, hit } = align(
    ru.map(([, n]) => item(segs, n).prose),
    ru.map(([u]) => u),
    du.map(([u]) => u)
  );
  const proseIds = ru.flatMap(([, n], i) => (item(segs, n).prose ? [i] : []));
  if (!proseIds.length || count(hit, proseIds) < SPLICE_MIN_LINE * proseIds.length) return today;
  const draftCps = Array.from(draft);
  const out: string[] = [];
  for (const [n, seg] of segs.entries()) {
    if (!seg.prose) {
      out.push(`$${seg.today}$`);
      continue;
    }
    const ids = proseIds.filter((i) => item(ru, i)[1] === n);
    const first = ids[0];
    const last = ids.at(-1);
    const lo = first === undefined ? 0 : item(start, first);
    const hi = last === undefined ? 0 : item(end, last);
    const sub = hi > lo ? draftCps.slice(item(du, lo)[1], item(du, hi - 1)[2]).join("") : "";
    if (!hasMoreLetters(sub, seg.today)) {
      out.push(seg.today); // nothing lost here: today's text, byte for byte
      continue;
    }
    const span = (hi - lo) / ids.length;
    if (count(hit, ids) < SPLICE_MIN_SEG * ids.length || !(span >= 0.5 && span <= 2.0))
      return today;
    const lead = seg.reading.slice(0, seg.reading.length - pyLstrip(seg.reading).length);
    const trail = seg.reading.slice(pyRstrip(seg.reading).length);
    out.push(lead + sub + trail);
  }
  return out.join("");
}
