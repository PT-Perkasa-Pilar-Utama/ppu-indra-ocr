// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/document.py (Apache-2.0, Copyright Datalab).

import type { Gray } from "./image.js";
import type { IndraPipeline } from "./pipeline.js";
import type { Box } from "./rec-preproc.js";
import type { TextLine } from "./textlayer.js";
import type { RecLine } from "./types.js";
import { item } from "./util.js";

const LETTER = /^\p{L}$/u;
const READABLE_MIN = 0.9; // share of the layer's letters the recognizer must know
const AGREE_LINES = 6; // text-layer lines whose script is checked against the page image
const AGREE_SHARE = 0.25; // under this share of agreeing lines, the layer is not the page's text
const LATIN_CODES = 0.9; // a layer line this Latin while its image is another script: legacy font codes
/** Page.note when the script check sends a page to OCR. */
export const NOT_THE_PAGE = "text layer does not match the page";

/** Share of the letters in `text` that are basic Latin (a-z, A-Z). */
export function latinShare(text: string): number {
  const letters = [...text].filter((c) => LETTER.test(c));
  return letters.filter((c) => c.charCodeAt(0) < 128).length / Math.max(letters.length, 1);
}

/**
 * Whether a text layer matches its page: at least AGREE_SHARE of the checked
 * lines agree. A legacy font (Kruti Dev and DevLys for Hindi, Bijoy for
 * Bangla) stores Latin codes for another script, so almost every line disagrees.
 */
export function layerAgrees(agree: readonly boolean[]): boolean {
  return !agree.length || agree.filter(Boolean).length >= AGREE_SHARE * agree.length;
}

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

/**
 * A line agrees unless its layer text is Latin while the script classifier
 * is sure its image is another script. Up to AGREE_LINES lines with four
 * letters or more are checked, spread over the page.
 */
export async function layerMatches(
  pipeline: IndraPipeline,
  gray: Gray,
  boxes: readonly Box[],
  rec: readonly (RecLine | undefined)[],
  t: Record<string, number>
): Promise<boolean> {
  const full = rec.flatMap((r, i) =>
    r && [...r.text].filter((c) => LETTER.test(c)).length >= 4 ? [i] : []
  );
  const step = Math.max(1, Math.floor(full.length / AGREE_LINES));
  const pick = full.filter((_, n) => n % step === 0).slice(0, AGREE_LINES);
  if (!pick.length) return true;
  const t0 = performance.now();
  const { names, sure } = await pipeline.boxGroups(
    gray,
    pick.map((i) => item(boxes, i))
  );
  const agree = pick.map(
    (i, n) =>
      !(latinShare(item(rec, i)?.text ?? "") >= LATIN_CODES && names[n] !== "base" && sure[n])
  );
  t["check_ms"] = performance.now() - t0;
  return layerAgrees(agree);
}
