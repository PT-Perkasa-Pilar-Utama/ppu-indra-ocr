// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
//
// Python 3.12 str semantics the table ports (table_rel.py, table_verify.py)
// depend on, where JavaScript differs: \s, html.escape and html.unescape.
// isdigit, the Latin-named letters, len, strip and split live in py-unicode.ts.
//
// Ceilings against Python (Unicode 15.0):
// - `\p{L}`, `\p{Uppercase}`, `\p{Lowercase}`, `\p{P}` and `\p{S}` come from the
//   engine's Unicode tables. They agree except for characters added after Unicode 15.0.
// - `pyHtmlUnescape` knows the named references amp, lt, gt, quot, apos and nbsp
//   (with and without the semicolon Python allows on the legacy names) and every
//   numeric reference. The other 2200 HTML5 names stay as written; the tables the
//   ports read hold only `pyHtmlEscape` output, which has none of them.

import { item } from "./util.js";

/** One letter: str.isalpha for one character. */
export const LETTER: RegExp = /^\p{L}$/u;
/** One uppercase character: str.isupper for one character. */
export const UPPER: RegExp = /^\p{Uppercase}$/u;
/** One lowercase character: str.islower for one character. */
export const LOWER: RegExp = /^\p{Lowercase}$/u;
/** Python's \s in a str pattern (the str.isspace characters), as the inside of a character class. */
export const PY_SPACE: string =
  "\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";

/** The character with these code points (source files stay ASCII). */
export const chars = (...cps: number[]): string => String.fromCodePoint(...cps);

/** Python's html.escape(s, quote=False). */
export function pyHtmlEscape(s: string): string {
  return s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

// cp1252 for the C1 range (html.unescape's _invalid_charrefs).
const C1 = [
  0x20ac, 0x81, 0x201a, 0x192, 0x201e, 0x2026, 0x2020, 0x2021, 0x2c6, 0x2030, 0x160, 0x2039, 0x152,
  0x8d, 0x17d, 0x8f, 0x90, 0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014, 0x2dc, 0x2122,
  0x161, 0x203a, 0x153, 0x9d, 0x17e, 0x178,
];
const NAMED = new Map<string, string>([
  ["amp", "&"],
  ["amp;", "&"],
  ["lt", "<"],
  ["lt;", "<"],
  ["gt", ">"],
  ["gt;", ">"],
  ["quot", '"'],
  ["quot;", '"'],
  ["apos;", "'"],
  ["nbsp", chars(0xa0)],
  ["nbsp;", chars(0xa0)],
]);
const CHARREF = /&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)/g;

/** html.unescape's numeric reference rules: a code point, or "" for one the standard drops. */
function numericRef(num: number): string {
  if (num === 0) return chars(0xfffd);
  if (num === 0x0d) return "\r";
  if (num >= 0x80 && num <= 0x9f) return chars(item(C1, num - 0x80));
  if ((num >= 0xd800 && num <= 0xdfff) || num > 0x10ffff) return chars(0xfffd);
  const dropped =
    (num >= 1 && num <= 8) ||
    num === 0xb ||
    (num >= 0xe && num <= 0x1f) ||
    num === 0x7f ||
    (num >= 0xfdd0 && num <= 0xfdef) ||
    (num >= 0xfffe && (num & 0xfffe) === 0xfffe);
  return dropped ? "" : chars(num);
}

/** Python's html.unescape for text made by `pyHtmlEscape`; see the ceiling in the file header. */
export function pyHtmlUnescape(s: string): string {
  if (!s.includes("&")) return s;
  return s.replace(CHARREF, (_, body: string) => {
    if (body.startsWith("#")) {
      const hex = body[1] === "x" || body[1] === "X";
      return numericRef(Number.parseInt(body.slice(hex ? 2 : 1).replace(/;$/, ""), hex ? 16 : 10));
    }
    const exact = NAMED.get(body);
    if (exact !== undefined) return exact;
    // the longest named reference that starts the text, as the standard says
    for (let x = body.length - 1; x > 1; x--) {
      const prefix = NAMED.get(body.slice(0, x));
      if (prefix !== undefined) return prefix + body.slice(x);
    }
    return `&${body}`;
  });
}
