// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/document.py (Apache-2.0, Copyright Datalab).

import { snippet } from "./agree.js";
import type { Gray } from "./image.js";
import type { LayerRead } from "./layer-gaps.js";
import type { IndraPipeline } from "./pipeline.js";
import { isAlpha } from "./py-unicode.js";
import type { Box } from "./rec-preproc.js";
import type { TextLayer, TextLine } from "./textlayer.js";
import type { RecLine, Tok } from "./types.js";
import { item } from "./util.js";

const LETTER = /^\p{L}$/u;
const READABLE_MIN = 0.9; // share of the layer's letters the recognizer must know
/** Text-layer lines whose start is read from the page image (agree.ts). */
export const CHECK_LINES = 6;
/** About this many characters from the start of each, cut at a word gap. */
export const CHECK_CHARS = 24;
/** A line's gap counts at most this much: one odd line cannot fail a page. */
const GAP_CAP = 10.0;
/** The first line read at most this far off: the layer matches, the rest is not read. */
export const EASY_GAP = 0.25;
/** A recognizer this sure of its own reading: a line it reads off the layer is strong evidence. */
const SURE_CONF = 0.97;
/**
 * The page score over the checked lines: a logistic model. Inputs, in order:
 * the trimmed mean gap, the median gap, the share of lines with a gap over
 * 0.5, 1, 2 and 4, the recognizers' mean confidence in their own readings,
 * per group recognizer (LAYER_GROUPS; the other lines are the base
 * recognizer's) the share of lines it read times the share of lines over 0.5,
 * and the share of lines over 1 that their recognizer reads with a confidence
 * of at least SURE_CONF. So lines that do not match count for more in a
 * script whose layers are often broken (Indic, Arabic) than in Latin, and a
 * line a sure recognizer reads otherwise counts for more than one it is
 * unsure of. Over 0, OCR reads the page.
 */
const LAYER_GROUPS = ["world", "rtl", "cjk"] as const;
const LAYER_WEIGHTS = [
  0.06462, 0.07808, -0.1493, 0.3135, 1.514, 1.05, 7.785, 2.993, 3.216, 5.219, 3.738,
] as const;
const LAYER_BIAS = -9.334;
/** Page.note when the check sends a page to OCR. */
export const NOT_THE_PAGE = "text layer does not match the page";

/**
 * Whether the recognizers know at least READABLE_MIN of the letters in these
 * runs. A recognizer that reads Hangul as jamo knows a syllable by its jamo.
 */
export function readable(pipeline: IndraPipeline, runs: readonly TextLine[]): boolean {
  const letters = runs.flatMap((r) => [...r.text].filter((c) => LETTER.test(c)));
  if (!letters.length) return true;
  const vocabs = pipeline.vocabs.map((v) => v.index);
  const known = letters.filter((c) =>
    vocabs.some((v) => v.has(c) || [...c.normalize("NFD")].every((j) => v.has(j)))
  ).length;
  return known >= READABLE_MIN * letters.length;
}

/** numpy's median of float64 values. */
function median(sorted: readonly number[]): number {
  const n = sorted.length;
  const mid = Math.floor(n / 2);
  return n % 2 ? item(sorted, mid) : (item(sorted, mid - 1) + item(sorted, mid)) / 2;
}

/**
 * The page score's inputs from the checked lines (a math line, gap null, is
 * left out); undefined when no line counts. Gaps are capped at GAP_CAP; the
 * trimmed mean leaves out the largest.
 */
export function layerFeatures(reads: readonly LayerRead[]): number[] | undefined {
  const lines = reads.flatMap((r) =>
    r.gap === null ? [] : [{ gap: Math.min(r.gap, GAP_CAP), conf: r.conf, group: r.group }]
  );
  if (!lines.length) return undefined;
  const gaps = lines.map((l) => l.gap).toSorted((a, b) => a - b);
  const kept = gaps.length > 2 ? gaps.slice(0, -1) : gaps;
  const n = gaps.length;
  const share = (pred: (g: number) => boolean): number => gaps.filter(pred).length / n;
  const over = share((g) => g > 0.5);
  return [
    kept.reduce((a, b) => a + b, 0) / kept.length,
    median(gaps),
    ...[0.5, 1.0, 2.0, 4.0].map((t) => share((g) => g > t)),
    lines.reduce((a, l) => a + l.conf, 0) / n,
    ...LAYER_GROUPS.map((name) => (over * lines.filter((l) => l.group === name).length) / n),
    lines.filter((l) => l.gap > 1.0 && l.conf >= SURE_CONF).length / n,
  ];
}

/**
 * Whether a text layer is its page's text: the page score over the checked
 * lines is at most 0. A legacy font or a broken ToUnicode map gives most
 * lines a gap of 2 to 20; a map that breaks only some letters (Thai tone
 * marks, Arabic lam-alef, Indic vowel signs) gives some lines a gap of 0.5 to
 * 3; a sound layer gives about 0, also where the recognizer is unsure.
 */
export function layerMatches(reads: readonly LayerRead[]): boolean {
  const x = layerFeatures(reads);
  if (!x) return true;
  return LAYER_BIAS + x.reduce((s, v, i) => s + (LAYER_WEIGHTS[i] ?? 0) * v, 0) <= 0;
}

/**
 * Whether the text layer is the page's text: the start of up to CHECK_LINES
 * of its lines is read from the page image (agree.ts). The line halfway down
 * goes first; the others only when it does not clearly match, or when the
 * layer dropped runs with bad codes (a font map that may break other runs too).
 */
export async function layerChecks(
  pipeline: IndraPipeline,
  gray: Gray,
  boxes: readonly Box[],
  toks: readonly Tok[][],
  rec: readonly (RecLine | undefined)[],
  tl: TextLayer,
  t: Record<string, number>
): Promise<boolean> {
  const t0 = performance.now();
  const full = rec.flatMap((r, i) =>
    r && Array.from(r.text).filter(isAlpha).length >= 4 ? [i] : []
  );
  const step = Math.max(1, Math.floor(full.length / CHECK_LINES));
  const pick = full.filter((_, n) => n % step === 0).slice(0, CHECK_LINES);
  if (!pick.length) return true;
  const first = Math.floor(pick.length / 2);
  const reads = await startGaps(pipeline, gray, boxes, toks, [item(pick, first)]);
  let ok = true;
  if (tl.stats.bad_runs || reads.some((r) => r.gap !== null && r.gap > EASY_GAP)) {
    reads.push(
      ...(await startGaps(
        pipeline,
        gray,
        boxes,
        toks,
        pick.filter((_, j) => j !== first)
      ))
    );
    ok = layerMatches(reads);
  }
  t["check_ms"] = performance.now() - t0;
  return ok;
}

/** The layer gaps of the start (CHECK_CHARS) of these detected lines. */
async function startGaps(
  pipeline: IndraPipeline,
  gray: Gray,
  boxes: readonly Box[],
  toks: readonly Tok[][],
  lines: readonly number[]
): Promise<LayerRead[]> {
  if (!lines.length) return [];
  const cut = lines.map((i) => snippet(item(toks, i), CHECK_CHARS));
  const sb = lines.map((i, n): Box => {
    const [x0, y0, x1, y1] = item(boxes, i);
    const x = item(cut, n).cut;
    return [x0, y0, x === undefined ? x1 : Math.fround(Math.min(x1, x)), y1];
  });
  return pipeline.layerGaps(
    gray,
    sb,
    cut.map((c) => c.toks)
  );
}
