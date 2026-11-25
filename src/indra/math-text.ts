// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/math.py (Apache-2.0, Copyright Datalab).
//
// The text rules around MathRec: which readings hold math, how the plain reading's
// accents and prose go back into a reading, and the learned heads that accept a
// reading and gate a read. No model runs here; the decoding is in math.ts.

import {
  ACCENT,
  SP,
  SPLICE_MIN_LINE,
  SPLICE_MIN_SEG,
  STYLE_SOURCE,
  TEXT_STYLE,
  fold,
  hasAccents,
  look,
  restoreAccents,
  spliceProse,
  units,
} from "./math-splice.js";
import type { Unit } from "./math-splice.js";
import {
  hasLatinName,
  isAlpha,
  isCategoryM,
  isDigit,
  isUpper,
  pyLen,
  pySplit,
} from "./py-unicode.js";
import { SequenceMatcher } from "./sequence-matcher.js";
import { item } from "./util.js";

// The alignment rules are in math-splice.ts; this file is the API of the text rules.
export { SPLICE_MIN_LINE, SPLICE_MIN_SEG, fold, look, restoreAccents, spliceProse, units };
export type { Unit };

/** A math reading replaces a line only at this mean token probability or more (no learned head). */
export const MATH_MIN_CONF = 0.9;

/** `math_accept.json`: a logistic head on the features of a reading. */
export type MathAcceptHead = {
  w: readonly number[];
  mu: readonly number[];
  sd: readonly number[];
  tau: number;
  features?: readonly string[];
};

/** `math_gate.json`: a logistic gate on `gateFeatures`. */
export type MathGateHead = {
  w: readonly number[];
  b: number;
  z_min: number;
  keep?: number;
  features?: string;
  train_reads?: number;
};

// Commands that style or space text, and escaped symbols (\%): no math of their own.
const STYLE = new RegExp(STYLE_SOURCE, "gu");

// Commands that write one symbol plain text has as well: a span of digits, letters
// and these is plain text ("66 +- 5").
const SYMBOL =
  /\\(?:pm|mp|times|cdot|div|leq|geq|le|ge|neq|ne|approx|sim|simeq|equiv|rightarrow|leftarrow|to|uparrow|downarrow|longrightarrow|circ|degree|ldots|cdots|dots|prime|bullet|star|ast|mid|vert|Rightarrow|Leftarrow|Leftrightarrow|leftrightarrow)(?![A-Za-z])/g;

const TEXT_STYLE_ARG = new RegExp(`\\\\(?:${TEXT_STYLE})[${SP}]*\\{([^{}]*)\\}`, "gu");
const COMMAND_ARG = new RegExp(`\\\\[A-Za-z]+[${SP}]*\\{`, "u");

const LATEX_COMMAND = /\\[A-Za-z]/;
const OUTSIDE_MARKUP = /\\[A-Za-z]|[\^_{}]/;
const COMMAND_OR_CHAR = /\\[A-Za-z]+/g;
const SIG_KEEP = new Set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=<>()/");

// Operators a draft may hold, and the currency signs: gate_features counts them.
const GATE_OPS = new Set(
  "=+-*/<>^_|"
    .split("")
    .concat(
      [
        0x2211, 0x222b, 0x221a, 0x2264, 0x2265, 0xb1, 0xd7, 0xf7, 0x2202, 0x221e, 0x2248, 0x2260,
        0x2208, 0x2200, 0x2203, 0x2192, 0x2190,
      ].map((cp) => String.fromCodePoint(cp))
    )
);
const CURRENCY = new Set(
  "$%".split("").concat([0x20ac, 0xa3, 0xa5].map((cp) => String.fromCodePoint(cp)))
);

const spans = (parts: readonly string[]): string[] => parts.filter((_, k) => k % 2 === 1);
const proseParts = (parts: readonly string[]): string[] => parts.filter((_, k) => k % 2 === 0);

/**
 * A MathRec reading holds math: a $...$ span with a command (\frac, \alpha, \leq),
 * a superscript or a subscript once styling and spacing are gone. A span of plain
 * characters (a number, a word, "x = 2") writes the same as plain text, and the line
 * keeps the recognizer's reading.
 */
export function hasMath(text: string): boolean {
  return spans(text.split("$")).some((span) => {
    const s = span.replace(SYMBOL, "").replace(STYLE, "");
    return s.includes("\\") || s.includes("^") || s.includes("_");
  });
}

/**
 * A reading's $...$ spans hold structure a text layer cannot write: a superscript, a
 * subscript, or a command with an argument (\frac{, \sqrt{, \hat{). A text layer has
 * every symbol (alpha, triangle, the element sign) as a character, so a span of
 * symbols and plain characters adds nothing to it.
 */
export function hasStructure(text: string): boolean {
  return spans(text.split("$")).some((span) => {
    const s = span.replace(TEXT_STYLE_ARG, "$1");
    return s.includes("^") || s.includes("_") || COMMAND_ARG.test(s);
  });
}

/**
 * The line's text when MathRec's reading replaces the plain reading (the recognizer's
 * or the text layer's), else null: the reading must hold math (hasMath) once accents
 * written as math are left out (when the plain reading has accented letters), and it
 * must not write LaTeX outside its $...$ spans (a malformed reading). The plain
 * reading's accents go back into the prose (restoreAccents).
 */
export function acceptMath(reading: string, draft: string): string | null {
  const parts = reading.split("$");
  if (proseParts(parts).some((p) => LATEX_COMMAND.test(p))) return null;
  const accented = hasAccents(draft);
  const kept = spans(parts).filter((sp) => !(accented && ACCENT.test(sp)));
  if (!hasMath(kept.map((sp) => `$${sp}$`).join(""))) return null;
  return restoreAccents(reading, draft);
}

/**
 * Whether MathRec may read a line with this text (the recognizer's or the text layer's
 * reading). MathRec writes Latin letters only: a line whose letters are mostly of
 * another script (Greek or Cyrillic prose, any other script) stays with its own
 * reading. A letter of another script on its own (a Greek variable) is a math symbol,
 * not text: only runs of 2 or more such letters count. A line without such text
 * (numbers, symbols) may go.
 */
export function latinLine(text: string): boolean {
  let latin = 0;
  let other = 0;
  let run = 0;
  for (const c of `${text} `) {
    if (isCategoryM(c)) {
      if (run > 0) run++; // a combining mark (Indic vowel sign) continues a word
      continue;
    }
    if (isAlpha(c) && !hasLatinName(c)) {
      run++;
      continue;
    }
    if (run >= 2) other += run;
    run = 0;
    if (isAlpha(c)) latin++;
  }
  return latin > other || other === 0;
}

/** Letters, digits and a few operators of a LaTeX reading (commands dropped). */
export const sigReading = (text: string): string =>
  Array.from(text.replace(COMMAND_OR_CHAR, " "))
    .filter((c) => SIG_KEEP.has(c))
    .join("");

/** The same for a plain reading, after NFKC. */
export const sigDraft = (text: string): string =>
  Array.from(text.normalize("NFKC"))
    .filter((c) => SIG_KEEP.has(c))
    .join("");

/**
 * Features of a flagged line for the read gate (mathGate): its math-classifier
 * probability, what its draft is made of, and page counts. 16 numbers.
 */
export function gateFeatures(
  prob: number,
  draft: string,
  nFlagged: number,
  pmax: number,
  nCand: number,
  source: string
): number[] {
  const cps = Array.from(draft);
  const n = Math.max(1, cps.length);
  const share = (f: (c: string) => boolean): number => cps.filter(f).length / n;
  return [
    prob,
    Math.log1p(cps.length),
    share(isDigit),
    share(isAlpha),
    share((c) => GATE_OPS.has(c)),
    share((c) => CURRENCY.has(c)),
    share((c) => c === " "),
    pySplit(draft).length,
    share((c) => 0x36f < (c.codePointAt(0) ?? 0) && (c.codePointAt(0) ?? 0) < 0x400),
    draft.includes("=") ? 1 : 0,
    share(isUpper),
    Math.log1p(nFlagged),
    pmax,
    Math.log1p(nCand),
    source === "ocr" ? 1 : 0,
    source === "text" ? 1 : 0,
  ];
}

/** The single-character tokens of `math_meta.json`: the letters MathRec can write. */
export const mathChars = (tokens: readonly string[]): ReadonlySet<string> =>
  new Set(tokens.filter((t) => pyLen(t) === 1));

/**
 * Whether MathRec's vocabulary writes this line's letters (the recognizer's or the
 * text layer's reading): at least `share` of them are its tokens. A MathRec whose
 * vocabulary holds another script's letters may then read lines of that script; one
 * with Latin letters only writes no such line, and latinLine alone decides.
 */
export function mathWrites(chars: ReadonlySet<string>, text: string, share = 0.9): boolean {
  const letterList = Array.from(text).filter(isAlpha);
  return (
    letterList.length > 0 &&
    letterList.filter((c) => chars.has(c)).length >= share * letterList.length
  );
}

/** The line's text with MathRec's prose splice (when on), for a reading that passed. */
function spliced(
  text: string,
  draft: string,
  out: string | null,
  proseSplice: boolean
): string | null {
  return out === null || !proseSplice ? out : spliceProse(text, draft, out);
}

/**
 * The line's text when MathRec's reading replaces the plain reading, else null. With
 * `math_accept.json` (`head`), a learned logistic head decides (features of the
 * reading, its confidence, the recognizer's confidence and the agreement with the
 * draft); else the reading needs conf >= MATH_MIN_CONF and acceptMath. A malformed
 * reading (LaTeX outside its $...$ spans) never replaces the line. With proseSplice,
 * the accepted text keeps the plain reading's prose (spliceProse).
 */
export function mathAccept(
  head: MathAcceptHead | null,
  text: string,
  conf: number,
  draft: string,
  rconf = 1.0,
  inTable = 0.0,
  proseSplice = true
): string | null {
  if (head === null) {
    const out = conf >= MATH_MIN_CONF ? acceptMath(text, draft) : null;
    return spliced(text, draft, out, proseSplice);
  }
  const parts = text.split("$");
  if (!text || proseParts(parts).some((p) => OUTSIDE_MARKUP.test(p))) return null;
  const kept = acceptMath(text, draft);
  const inside =
    spans(parts).reduce((n, p) => n + pyLen(p), 0) / Math.max(1, pyLen(text.replaceAll("$", "")));
  const agree = new SequenceMatcher(sigDraft(draft), sigReading(text)).ratio();
  const ok = kept === null ? 0 : 1;
  const x = [conf, rconf, ok, inside, agree, Math.log1p(pySplit(text).length), ok * conf];
  if (head.mu.length > x.length) x.push(inTable); // the head also knows whether the layout put the line in a table
  // Python's zip stops at the shortest list
  const n = Math.min(head.w.length - 1, x.length, head.mu.length, head.sd.length);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += (item(head.w, i + 1) * (item(x, i) - item(head.mu, i))) / item(head.sd, i);
  }
  const z = item(head.w, 0) + sum;
  if (1.0 / (1.0 + Math.exp(-z)) < head.tau) return null;
  return spliced(text, draft, kept ?? (text.includes("$") ? text : null), proseSplice);
}

/**
 * Which flagged lines MathRec reads. With `math_gate.json` (`head`), a learned logistic
 * gate skips lines whose reading would not be used (features: gateFeatures); without
 * it every flagged line is read.
 */
export function mathGate(
  head: MathGateHead | null,
  probs: readonly number[],
  drafts: readonly string[],
  nCand: number,
  source: string
): boolean[] {
  if (head === null || probs.length === 0) return probs.map(() => true);
  const pmax = Math.max(...probs);
  return probs.slice(0, drafts.length).map((p, i) => {
    const x = gateFeatures(p, item(drafts, i), probs.length, pmax, nCand, source);
    const n = Math.min(head.w.length, x.length);
    let z = 0;
    for (let k = 0; k < n; k++) z += item(head.w, k) * item(x, k);
    return head.b + z >= head.z_min;
  });
}

/** The decoder's state after a token, and whether the reading can no longer pass acceptMath. */
export type TextLikeStep = { inMath: boolean; textLike: boolean };

/**
 * Tracks one emitted token: true when the reading can no longer pass acceptMath (a
 * LaTeX command outside the $...$ spans). `inMath` is whether the decoder is inside a
 * $...$ span before the token; the result has it after the token.
 */
export function textLike(token: string, inMath: boolean): TextLikeStep {
  if (token === "$") return { inMath: !inMath, textLike: false };
  const cps = Array.from(token);
  return {
    inMath,
    textLike: !inMath && cps.length > 1 && cps[0] === "\\" && isAlpha(item(cps, 1)),
  };
}
