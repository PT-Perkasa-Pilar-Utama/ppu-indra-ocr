// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
//
// Python's str and unicodedata rules for one code point, as the reference
// (Python 3.12, Unicode 15.0.0) applies them. JavaScript has the same rule for
// isalpha, isupper, lower, normalize and the M categories, so those call the
// engine. It has no API for isdigit (Nd plus superscripts and circled digits),
// canonical combining class, or the "LATIN" name prefix, so those read the
// range table below, made by scripts/golden/math-text.py --tables.
//
// Ceiling: the engine rules (alpha, upper, lower, NFKD, NFKC, category M) follow
// the engine's Unicode version. They equal Python 3.12 for every code point
// that Unicode 15.0 assigns (tests/indra/py-unicode.test.ts hashes all of them);
// a code point added after 15.0 follows the engine, where Python 3.12 treats it
// as unassigned. The generated tables follow 15.0 only.

import { isPySpace } from "./util.js";

const ALPHA = /^\p{L}$/u;
const UPPER = /^\p{Uppercase}$/u;
const CATEGORY_M = /^\p{M}$/u;

/** Table flags. */
const COMBINING = 1;
const LATIN_NAME = 2;
const DIGIT = 4;

// BEGIN GENERATED (scripts/golden/math-text.py --tables)
// Unicode 15.0.0, Python 3.12.10.
/** "gap len flags" triples in base 36: ranges of code points with a table flag. */
const FLAG_TABLE: string =
  "1c 9 4 7 p 2 6 p 2 1j 1 4 5 0 4 6 m 2 1 u 2 1 c7 2 28 26 1 1 v 1 7n 4 1 7d 18 1 1 0 1 1 " +
  "1 1 1 1 1 1 0 1 20 a 1 1c k 1 0 9 4 6 0 1 2t 6 1 2 5 1 2 1 1 1 3 1 2 9 4 n 0 1 u q 1 39 " +
  "9 4 x 8 1 9 0 1 o 3 1 1 8 1 1 2 1 1 4 1 17 2 1 1o 7 1 16 n 1 1 s 1 1o 0 1 g 0 1 3 3 1 h " +
  "9 4 24 0 1 g 0 1 o 9 4 e 0 1 1p 0 1 g 0 1 o 9 4 24 0 1 g 0 1 o 9 4 24 0 1 g 0 1 o 9 4 2l" +
  " 0 1 o 9 4 24 0 1 g 0 1 7 1 1 f 9 4 24 0 1 g 0 1 o 9 4 23 1 1 g 0 1 o 9 4 2i 0 1 r 9 4 2" +
  "0 2 1 d 3 1 4 9 4 2m 2 1 d 3 1 4 9 4 1q 1 1 6 9 4 b 0 1 1 0 1 1 0 1 1j 1 1 1 0 1 5 3 1 2" +
  " 0 1 1 2 1 1 1 1 1q 0 1 34 0 1 1 1 1 5 9 4 1v 0 1 2 9 4 jn 2 1 9 8 4 pu 1 1 u 0 1 4d 0 1" +
  " a 0 1 2 9 4 12 9 4 3z 0 1 3z 2 1 a 9 4 3k a 4 1o 1 1 1z 0 1 k 7 1 2 0 1 0 9 4 6 9 4 m d" +
  " 1 1 f 1 2t 0 1 f 0 1 b 9 4 h 8 1 1i 1 1 4 9 4 18 0 1 b 1 1 1v 0 1 8 9 4 6 9 4 3a 2 1 1 " +
  "c 1 1 6 1 4 0 1 6 0 1 3 1 1 6 11 2 1o 3 2 5 c 2 1 x 2 11 1r 1 0 73 2 a8 0 4 3 5 4 6 9 4 " +
  "6 c 2 1f c 1 4 0 1 3 b 1 43 0 2 kb 8 4 b 8 4 b 8 4 2h 0 4 a 8 4 1 0 4 f1 0 2 2g 8 4 1 8 " +
  "4 1 8 4 y5 s 2 1 1 2 33 2 1 3x 0 1 2o v 1 fe 5 1 2x 1 1 n7p 9 4 1x 0 1 4 9 1 w 1 1 28 1 " +
  "1 1c 25 2 1 m 2 3 1r 2 5 1 2 1 0 2 1 4 2 r 2 2 2 5 2 6 0 1 11 0 1 47 0 1 b 9 4 6 h 1 e 9" +
  " 4 x 2 1 11 0 1 2n 0 1 c 0 1 f 9 4 m 9 4 2e 9 4 2e 0 1 1 2 1 2 1 1 5 1 1 1 0 1 1g 0 1 1l" +
  " 16 2 5 4 2 1 2 2 3o 0 1 2 9 4 fly 6 2 n 0 1 ld f 1 68 9 4 kj 0 1 6a 0 1 45 4 1 85 9 4 1" +
  "2b 0 1 1 0 1 14 2 1 4 0 1 0 3 4 4h 1 1 fx 3 1 8 9 4 86 8 4 1u 1 1 28 2 1 1y a 1 1d 3 1 5" +
  "c 0 1 b 8 4 b 9 4 0 0 1 e 0 1 1l 1 1 1h 9 4 6 2 1 1c 1 1 1 9 4 1f 0 1 24 0 1 9 0 1 5 9 4" +
  " 2j 1 1 4y 1 1 5 9 4 1t 1 1 g 0 1 o 6 1 3 4 1 5p 0 1 3 0 1 9 9 4 4 0 1 2r 1 1 c 9 4 6d 1" +
  " 1 3i 0 1 g 9 4 2k 1 1 8 9 4 2p 0 1 4 9 4 73 1 1 4l 9 4 2b 1 1 4 0 1 c 9 4 3q 0 1 2b 0 1" +
  " i 0 1 29 0 1 bp 0 1 g 9 4 6g 0 1 1 1 1 a 9 4 1p 0 1 8 9 4 bb 1 1 d 9 4 eti 9 4 2e 9 4 1" +
  "2 4 1 1n 6 1 p 9 4 wm 1 1 f58 0 1 43q 4 1 3 5 1 8 7 1 2 6 1 u 3 1 44 2 1 13d 1d 4 1ds u " +
  "2 6 5 2 5x 6 1 1 g 1 2 6 1 1 1 1 1 4 1 2s 0 1 4g 6 1 9 9 4 9w 0 1 1p 3 1 0 9 4 du 3 1 0 " +
  "9 4 ra 6 1 31 6 1 5 9 4 1ie a 4 25h 9 4";
/** "name:code point" pairs of the single-word GREEK SMALL LETTER names. */
const GREEK_TABLE: string =
  "heta:371 alpha:3b1 beta:3b2 gamma:3b3 delta:3b4 epsilon:3b5 zeta:3b6 eta:3b7 theta:3b8 iota:3b9 kappa:3ba lamda:3bb mu:3bc nu:3bd xi:3be omicron:3bf pi:3c0 rho:3c1 sigma:3c3 tau:3c4 upsilon:3c5 phi:3c6 chi:3c7 psi:3c8 omega:3c9 stigma:3db digamma:3dd koppa:3df sampi:3e1 sho:3f8 san:3fb";
// END GENERATED

type FlagRanges = { starts: Uint32Array; ends: Uint32Array; flags: Uint8Array };

function decodeFlags(table: string): FlagRanges {
  const t = table.split(" ").map((s) => Number.parseInt(s, 36));
  const n = t.length / 3;
  const starts = new Uint32Array(n);
  const ends = new Uint32Array(n);
  const flags = new Uint8Array(n);
  let end = -1;
  for (let i = 0; i < n; i++) {
    const start = end + 1 + (t[3 * i] ?? 0);
    end = start + (t[3 * i + 1] ?? 0);
    starts[i] = start;
    ends[i] = end;
    flags[i] = t[3 * i + 2] ?? 0;
  }
  return { starts, ends, flags };
}

let ranges: FlagRanges | undefined;

function flagsOf(c: string): number {
  ranges ??= decodeFlags(FLAG_TABLE);
  const cp = c.codePointAt(0) ?? 0;
  let lo = 0;
  let hi = ranges.starts.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cp < (ranges.starts[mid] ?? 0)) hi = mid - 1;
    else if (cp > (ranges.ends[mid] ?? 0)) lo = mid + 1;
    else return ranges.flags[mid] ?? 0;
  }
  return 0;
}

let greek: Map<string, string> | undefined;

/** str.isalpha for one code point: categories Lm, Lt, Lu, Ll, Lo. */
export const isAlpha = (c: string): boolean => ALPHA.test(c);

/** str.isupper for one code point. */
export const isUpper = (c: string): boolean => UPPER.test(c);

/** str.isdigit for one code point. */
export const isDigit = (c: string): boolean => (flagsOf(c) & DIGIT) !== 0;

/** unicodedata.category(c).startswith("M"): a combining mark of any kind. */
export const isCategoryM = (c: string): boolean => CATEGORY_M.test(c);

/** unicodedata.combining(c) != 0: a nonzero canonical combining class. */
export const isCombining = (c: string): boolean => (flagsOf(c) & COMBINING) !== 0;

/** unicodedata.name(c, "").startswith("LATIN"). */
export const hasLatinName = (c: string): boolean => (flagsOf(c) & LATIN_NAME) !== 0;

/**
 * unicodedata.lookup("GREEK SMALL LETTER " + name.upper()), or undefined where
 * Python raises KeyError. Only the one-word names resolve ("lambda" does not:
 * the Unicode name is LAMDA).
 */
export function greekSmallLetter(name: string): string | undefined {
  greek ??= new Map(
    GREEK_TABLE.split(" ").map((p): [string, string] => {
      const [word = "", hex = "0"] = p.split(":");
      return [word, String.fromCodePoint(Number.parseInt(hex, 16))];
    })
  );
  return greek.get(name);
}

/** Python's len(s): code points, not UTF-16 units. */
export const pyLen = (s: string): number => Array.from(s).length;

/** str.split() with no argument: runs of Python whitespace separate the words. */
export function pySplit(s: string): string[] {
  const words: string[] = [];
  let word = "";
  for (const c of s) {
    if (isPySpace(c)) {
      if (word) words.push(word);
      word = "";
    } else {
      word += c;
    }
  }
  if (word) words.push(word);
  return words;
}

/** str.lstrip() with no argument. */
export function pyLstrip(s: string): string {
  let i = 0;
  while (i < s.length && isPySpace(s.charAt(i))) i++;
  return s.slice(i);
}

/** str.rstrip() with no argument. */
export function pyRstrip(s: string): string {
  let n = s.length;
  while (n > 0 && isPySpace(s.charAt(n - 1))) n--;
  return s.slice(0, n);
}

/** str.strip() with no argument. */
export const pyStrip = (s: string): string => pyLstrip(pyRstrip(s));
