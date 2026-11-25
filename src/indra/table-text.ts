// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/table_rel.py (Apache-2.0, Copyright Datalab).
//
// The word text inputs of tsr_rel.onnx (hashed word ids) and tsr_refine.onnx
// (content features). The models expect exactly these values.
//
// Ceiling against Python: NFKC and case folding come from the engine's Unicode
// tables, which agree with Python's (Unicode 15.0) except for newer characters.

import { LETTER, LOWER, PY_SPACE, UPPER, chars } from "./py-text.js";
import { hasLatinName, isDigit, pyLen, pyStrip } from "./py-unicode.js";
import { item } from "./util.js";

export const N_REFINE_TEXT = 13;

const DECIMAL = /\p{Nd}/gu;

const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

/** zlib.crc32 of a byte string. */
export function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (const b of bytes) c = item(CRC_TABLE, (c ^ b) & 0xff) ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const utf8 = new TextEncoder();

/**
 * Hashed word ids in [1, nVocab) (lower case NFKC, every digit 0). A word with a
 * letter outside the Latin script gets id 0 (no word embedding).
 */
export function wordIds(texts: readonly string[], nVocab: number): number[] {
  return texts.map((t) => {
    const s = t.normalize("NFKC").toLowerCase().replace(DECIMAL, "0");
    for (const c of s) if (LETTER.test(c) && !hasLatinName(c)) return 0;
    return (crc32(utf8.encode(s)) % (nVocab - 1)) + 1;
  });
}

// A number: optional bracket, sign, currency sign, bracket, then digits, digit
// separators and spaces, closing marks and footnote stars.
const REF_NUM = new RegExp(
  `^[(\\[]?[-+${chars(0x2212, 0x2013)}]?[$${chars(0x20ac, 0xa3, 0xa5)}]?[(\\[]?\\p{Nd}[\\p{Nd},.${PY_SPACE}]*[%)\\]]*\\*{0,3}$`,
  "u"
);
const REF_DASH = new Set([
  "-",
  chars(0x2013),
  chars(0x2014),
  chars(0x2212),
  "n/a",
  "na",
  "nd",
  "n.a.",
  "nr",
  chars(0x2026),
  "...",
  chars(0xb7),
]);
const REF_MARK = new Set([
  "*",
  "**",
  "***",
  chars(0x2020),
  chars(0x2021),
  chars(0xa7),
  chars(0xb6),
  "#",
]);
const CURRENCY = chars(0x24, 0x20ac, 0xa3, 0xa5);

/**
 * [N, N_REFINE_TEXT] content features of each word for tsr_refine.onnx: has a
 * digit, digit share, letter share, number form, first letter upper, all upper,
 * length, ends with ':', has '%', has a currency sign, has a bracket, a dash or
 * n/a, a footnote mark.
 */
export function refineTextFeatures(texts: readonly string[]): Float32Array {
  const out = new Float32Array(texts.length * N_REFINE_TEXT);
  for (const [i, t] of texts.entries()) {
    const s = pyStrip(t.normalize("NFKC"));
    if (!s) continue;
    const n = pyLen(s);
    const cps = [...s];
    const nd = cps.filter(isDigit).length;
    const letters = cps.filter((c) => LETTER.test(c));
    const row = out.subarray(i * N_REFINE_TEXT, (i + 1) * N_REFINE_TEXT);
    row[0] = nd > 0 ? 1 : 0;
    row[1] = nd / n;
    row[2] = letters.length / n;
    row[3] = REF_NUM.test(s) ? 1 : 0;
    row[4] = letters.length > 0 && UPPER.test(item(letters, 0)) ? 1 : 0;
    row[5] = letters.length >= 2 && letters.every((c) => UPPER.test(c)) ? 1 : 0;
    row[6] = Math.log1p(n) / 3;
    row[7] = s.endsWith(":") ? 1 : 0;
    row[8] = s.includes("%") ? 1 : 0;
    row[9] = cps.some((c) => CURRENCY.includes(c)) ? 1 : 0;
    row[10] = cps.some((c) => "()[]".includes(c)) ? 1 : 0;
    row[11] = REF_DASH.has(s.toLowerCase()) ? 1 : 0;
    row[12] = REF_MARK.has(s) || (n === 1 && LETTER.test(s) && LOWER.test(s)) ? 1 : 0;
  }
  return out;
}
