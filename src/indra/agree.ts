// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/agree.py (Apache-2.0, Copyright Datalab).

import { sumF64 } from "./numeric.js";
import { isAlpha, isCategoryM } from "./py-unicode.js";
import { flip, hasRtl, normalize as rtlNormalize } from "./rtl.js";
import type { Tok } from "./types.js";
import { isPySpace, item } from "./util.js";
import type { Vocab } from "./vocab.js";

/**
 * Does a PDF text layer say what its page shows? A legacy font (Kruti Dev,
 * Bijoy, VNI) stores Latin codes for another script, a broken ToUnicode map
 * writes other letters, and a layer can hold syllables in visual order; the
 * ink checks in textlayer.ts do not see this. The check reads a few short
 * pieces of layer lines from the page image and asks how likely each
 * recognizer finds the layer's text (a forced CTC alignment) compared with its
 * own best reading. The gap is near 0 where the layer is the page's text, also
 * where the recognizer is unsure, and large where the layer's characters are
 * not in the image.
 */

/** The class ids that may write one layer character; empty for one the recognizer cannot write. */
export type TargetAlts = readonly number[];

/** One line's recognizer outputs and the layer text's target ids, for forcedGaps. */
export type GapItem = {
  /** The blank's log-probability per frame [T]. */
  blank: ArrayLike<number>;
  /** The top-k classes' log-probabilities [T, k], best first. */
  topv: ArrayLike<number>;
  topi: ArrayLike<number | bigint>;
  k: number;
  ids: readonly TargetAlts[];
};

/** The start of a line's layer tokens, and the x where it is cut (undefined: the whole line). */
export type Snippet = { toks: Tok[]; cut?: number };

/** Python's string order (code units here; they agree on the BMP). */
function cmpStr(a: string, b: string): number {
  if (a === b) return 0;
  return a < b ? -1 : 1;
}

/**
 * The start (leftmost by x) of a detected line's layer tokens: about `n`
 * non-space characters, cut at the first word gap after them when one comes
 * within 2n characters. Returns those tokens and the cut x (undefined: the
 * whole line).
 */
export function snippet(tokens: readonly Tok[], n: number): Snippet {
  const chars = tokens
    .filter((t) => !isPySpace(String(t[0])))
    .map((t): [number, string] => [t[1], String(t[0])])
    .toSorted((a, b) => a[0] - b[0] || cmpStr(a[1], b[1]));
  const xs = chars.map(([x]) => x);
  if (xs.length <= n) return { toks: [...tokens] };
  const spaces = tokens.filter((t) => isPySpace(String(t[0]))).map((t) => t[1]);
  let k = n;
  for (let j = n; j < Math.min(xs.length, 2 * n); j++) {
    if (spaces.some((s) => item(xs, j - 1) <= s && s < item(xs, j))) {
      k = j;
      break;
    }
  }
  // Never cut before a combining mark: a vowel sign or tone mark is drawn on
  // the letter before it, so the image piece shows it.
  while (k < xs.length && isCategoryM(Array.from(item(chars, k)[1])[0] ?? " ")) k++;
  if (k >= xs.length) return { toks: [...tokens] };
  const cut = (item(xs, k - 1) + item(xs, k)) / 2;
  return { toks: tokens.filter((t) => t[1] < cut), cut };
}

/**
 * The text of layer tokens as a recognizer reads the line: each word (a
 * text-layer run's characters between spaces) keeps the layer's order, and
 * the words go left to right on the line (right to left for a recognizer of
 * right-to-left text, which reads the logical order flipped). A layer can
 * store a line's words in another order than they stand, so the order of the
 * layer's words is no evidence against it.
 */
export function layerText(tokens: readonly Tok[], rtl: boolean): string {
  const words: Tok[][] = [];
  for (const t of tokens) {
    if (isPySpace(String(t[0]))) {
      if (words.at(-1)?.length) words.push([]);
      continue;
    }
    const last = words.at(-1);
    if (!last || (last.length && item(last, 0)[2] !== t[2])) words.push([t]);
    else last.push(t);
  }
  const kept = words.filter((w) => w.length);
  const key = (w: Tok[]): number =>
    rtl ? -Math.max(...w.map((t) => t[1])) : Math.min(...w.map((t) => t[1]));
  return kept
    .toSorted((a, b) => key(a) - key(b))
    .map((w) => w.map((t) => String(t[0])).join(""))
    .join(" ");
}

/** Whether a recognizer writes a character, Hangul by its jamo when the vocab holds jamo. */
function knows(vocab: Vocab, c: string): boolean {
  return (
    vocab.index.has(c) ||
    (vocab.jamo && Array.from(c.normalize("NFD")).every((j) => vocab.index.has(j)))
  );
}

/** Whether a recognizer can write at least `share` of the text's letters. */
export function writes(text: string, vocab: Vocab, share = 0.9): boolean {
  const letters = Array.from(vocab.normalize(text)).filter(isAlpha);
  if (!letters.length) return true;
  return letters.filter((c) => knows(vocab, c)).length >= share * letters.length;
}

/** The letter a precomposed character is built on (a letter and combining marks), else undefined. */
export function baseLetter(c: string): string | undefined {
  const d = Array.from(c.normalize("NFD"));
  const [first, ...marks] = d;
  return first !== undefined && marks.length && isAlpha(first) && marks.every(isCategoryM)
    ? first
    : undefined;
}

/**
 * The recognizer's class ids for a layer text: per token, the ids that write
 * it (the character in either case: small capitals and text set in capitals
 * draw another case than the layer stores; and a precomposed letter's base
 * letter: a recognizer that misses a mark the layer has is no evidence
 * against the layer), empty for a character the recognizer cannot write.
 * `visual`: the recognizer reads right-to-left text in visual order.
 */
export function targetIds(text: string, vocab: Vocab, visual: boolean): TargetAlts[] {
  let t = hasRtl(text) ? rtlNormalize(text) : text;
  t = vocab.normalize(t);
  if (visual) t = flip(t);
  const ids: TargetAlts[] = [];
  for (const c of vocab.tokenize(t)) {
    const forms = [c, c.toUpperCase(), c.toLowerCase()];
    const b = baseLetter(c);
    if (b) forms.push(b, b.toUpperCase(), b.toLowerCase());
    const alts = [...new Set(forms.flatMap((x) => vocab.index.get(x) ?? []))];
    const jamo = Array.from(c.normalize("NFD"));
    if (alts.length) {
      ids.push(alts);
    } else if (vocab.jamo && jamo.every((j) => vocab.index.has(j))) {
      for (const j of jamo) ids.push([vocab.index.get(j) ?? 0]); // Hangul as jamo
    } else {
      ids.push([]);
    }
  }
  return ids;
}

/** The log-probability a frame gives token `alts`: its best top-k class, or 1 nat under the k-th. */
function tokenLogp(it: GapItem, t: number, alts: TargetAlts): number {
  let best = -Infinity;
  let hit = false;
  for (let kk = 0; kk < it.k; kk++) {
    const cls = Number(it.topi[t * it.k + kk] ?? -1);
    if (!alts.includes(cls)) continue;
    hit = true;
    best = Math.max(best, Number(it.topv[t * it.k + kk] ?? -Infinity));
  }
  return hit ? best : Number(it.topv[t * it.k + it.k - 1] ?? -Infinity) - 1;
}

/**
 * Per item: the log-probability per token by which the best CTC alignment of
 * its ids falls below the recognizer's best path. A class outside the top k
 * counts 1 nat under the k-th, as does a character the recognizer cannot
 * write. 0 for no ids, Infinity when the ids cannot fit the frames.
 *
 * The reference aligns every item in one padded batch; the path sums run over
 * the batch's longest frame count (padding adds zeros), so the sums here do
 * too: numpy's pairwise sum rounds by the array's length.
 */
export function forcedGaps(items: readonly GapItem[]): number[] {
  const out = items.map(() => 0);
  const run = items.flatMap((it, i) => (it.ids.length && it.blank.length ? [i] : []));
  if (!run.length) return out;
  const T = Math.max(...run.map((i) => item(items, i).blank.length));
  for (const i of run) {
    const it = item(items, i);
    const n = it.blank.length;
    const L = it.ids.length;
    const S = 2 * L + 1;
    // A token may follow the one before it directly (no blank between) unless the two share a class.
    const skip = new Uint8Array(S);
    for (let j = 1; j < L; j++) {
      const prev = new Set(item(it.ids, j - 1));
      skip[2 * j + 1] = item(it.ids, j).some((c) => prev.has(c)) ? 0 : 1;
    }
    const lp = (t: number, s: number): number =>
      s % 2 === 0 ? Number(it.blank[t]) : tokenLogp(it, t, item(it.ids, (s - 1) / 2));
    let a = new Float64Array(S).fill(-Infinity);
    a[0] = lp(0, 0);
    if (S > 1) a[1] = lp(0, 1);
    let best = -Infinity;
    if (n - 1 === 0) best = Math.max(a[2 * L] ?? -Infinity, a[2 * L - 1] ?? -Infinity);
    for (let t = 1; t < n; t++) {
      const nxt = a.slice();
      for (let s = 1; s < S; s++) nxt[s] = Math.max(nxt[s] ?? -Infinity, a[s - 1] ?? -Infinity);
      for (let s = 2; s < S; s++) {
        if (skip[s]) nxt[s] = Math.max(nxt[s] ?? -Infinity, a[s - 2] ?? -Infinity);
      }
      for (let s = 0; s < S; s++) nxt[s] = (nxt[s] ?? -Infinity) + lp(t, s);
      a = nxt;
      if (t === n - 1) best = Math.max(a[2 * L] ?? -Infinity, a[2 * L - 1] ?? -Infinity);
    }
    const top = new Float64Array(T);
    for (let t = 0; t < n; t++) {
      const v = Number(it.topv[t * it.k] ?? -Infinity);
      top[t] = Number.isFinite(v) ? v : 0;
    }
    out[i] = Number.isFinite(best) ? (sumF64(top) - best) / L : Infinity;
  }
  return out;
}
