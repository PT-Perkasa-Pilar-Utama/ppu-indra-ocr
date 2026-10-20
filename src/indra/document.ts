// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/document.py (Apache-2.0, Copyright Datalab).

import type { Gray } from "./image.js";
import { decodeToks } from "./lines.js";
import { MATH_MIN_CONF } from "./math.js";
import type { IndraPipeline } from "./pipeline.js";
import type { Box } from "./rec-preproc.js";
import { cropLine } from "./rec-preproc.js";
import type { PdfPageSource, TextLayer, TextLine } from "./textlayer.js";
import { BROKEN_FONT_MAP, uprightTextLayer } from "./textlayer.js";
import type { Page, RecLine, Tok } from "./types.js";
import { isStrongL } from "./unicode.js";
import { isRtl } from "./rtl.js";
import { rot90 } from "./upright.js";
import { isPySpace, item } from "./util.js";

const CHECK_DPI = 72; // resolution of the render for the text-layer ink checks
const MAX_PAGE_PIXELS = 100e6; // a larger page renders at a lower dpi
const READABLE_MIN = 0.9; // share of the layer's letters the recognizer must know
const LETTER = /^\p{L}$/u;

/** auto: text layer where sound, OCR elsewhere. ocr: always OCR. text: the text layer even when the checks fail. */
export type PdfMode = "auto" | "ocr" | "text";

export type PdfPageOptions = {
  /** Default "auto". */
  mode?: PdfMode;
  /** Render resolution for OCR. Default 200. */
  dpi?: number;
  /** OCR detected lines that hold no text-layer characters (text inside images). Default true. */
  imageOcr?: boolean;
};

/** Tokens per detected line, and the characters outside every line. */
export type AssignedRuns = { toks: Tok[][]; lost: number };

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

/** Unicode bidi base direction: most strong characters are right-to-left. */
export function rightToLeft(tokens: readonly Tok[]): boolean {
  let rtl = 0;
  let ltr = 0;
  for (const [c] of tokens) {
    if (typeof c !== "string" || [...c].length !== 1) continue;
    if (isRtl(c)) rtl++;
    else if (isStrongL(c)) ltr++;
  }
  return rtl > ltr;
}

/** Whether more than `frac` of `box` lies inside one of `boxes`. */
function covered(box: Box, boxes: readonly Box[], frac = 0.5): boolean {
  const f = Math.fround;
  const area = Math.max(f(f(box[2] - box[0]) * f(box[3] - box[1])), 1e-6);
  return boxes.some((b) => {
    const ix = Math.max(f(Math.min(box[2], b[2]) - Math.max(box[0], b[0])), 0);
    const iy = Math.max(f(Math.min(box[3], b[3]) - Math.max(box[1], b[1])), 0);
    return f(f(ix * iy) / area) > frac;
  });
}

/**
 * Text-layer characters per detected line (`boxes` in pixels, `k` pixels per
 * point). Each character goes to the smallest box that contains its center; a
 * space follows the character before it. Inside a line, each run's part keeps
 * the PDF order, and the parts are ordered by x (right to left in an RTL line).
 */
export function assignRuns(
  runs: readonly TextLine[],
  boxes: readonly Box[],
  k: number
): AssignedRuns {
  if (!boxes.length) {
    return {
      toks: [],
      lost: runs.reduce((n, r) => n + r.chars.filter(([c]) => !isPySpace(c)).length, 0),
    };
  }
  const f = Math.fround;
  const area = boxes.map(([x0, y0, x1, y1]) => f(f(x1 - x0) * f(y1 - y0)));
  // line -> run index (first appearance order) -> tokens
  const parts = boxes.map(() => new Map<number, Tok[]>());
  let lost = 0;
  for (const run of runs) {
    const idx = run.seq;
    let j: number | undefined;
    for (const [c, b] of run.chars) {
      if (!isPySpace(c)) {
        const cx = ((b[0] + b[2]) / 2) * k;
        const cy = ((b[1] + b[3]) / 2) * k;
        // numpy compares the float32 boxes with the center rounded to float32.
        const fx = f(cx);
        const fy = f(cy);
        let best: number | undefined;
        for (const [i, [x0, y0, x1, y1]] of boxes.entries()) {
          if (
            x0 <= fx &&
            fx <= x1 &&
            y0 <= fy &&
            fy <= y1 &&
            (best === undefined || item(area, i) < item(area, best))
          ) {
            best = i;
          }
        }
        j = best;
        if (j === undefined) {
          lost++;
          continue;
        }
        const line = item(parts, j);
        line.set(idx, [...(line.get(idx) ?? []), [c, cx, idx]]);
      } else if (j !== undefined) {
        const part = item(parts, j).get(idx) ?? [];
        part.push([c, part.at(-1)?.[1] ?? 0, idx]);
      }
    }
  }
  const toks = parts.map((line) => {
    const all = [...line.values()].flat();
    const rtl = rightToLeft(all);
    const key = (toksOf: Tok[]): number =>
      rtl ? -Math.max(...toksOf.map((t) => t[1])) : Math.min(...toksOf.map((t) => t[1]));
    const order = [...line.keys()].toSorted(
      (a, b) => key(line.get(a) ?? []) - key(line.get(b) ?? [])
    );
    const tk: Tok[] = [];
    for (const idx of order) {
      const last = tk.at(-1);
      if (last) tk.push([" ", last[1], idx]); // parts of different runs
      tk.push(...(line.get(idx) ?? []));
    }
    return tk;
  });
  return { toks, lost };
}

/** Pixels per point for a page: the dpi, lowered for pages past MAX_PAGE_PIXELS. */
function pageScale(page: PdfPageSource, dpi: number): number {
  const [w, h] = page.size();
  return Math.min(dpi / 72, Math.sqrt(MAX_PAGE_PIXELS / Math.max(w * h, 1)));
}

function timed<T>(t: Record<string, number>, key: string, run: () => T): T {
  const t0 = performance.now();
  const out = run();
  t[key] = performance.now() - t0;
  return out;
}

/** One PDF page: the text layer when it can be trusted, OCR otherwise. */
export async function pdfPage(
  pipeline: IndraPipeline,
  page: PdfPageSource,
  options: PdfPageOptions = {}
): Promise<Page> {
  const { mode = "auto", dpi = 200, imageOcr = true } = options;
  const t: Record<string, number> = {};
  let note = "mode=ocr";
  let turn = true;
  if (mode !== "ocr") {
    const s = CHECK_DPI / 72;
    const { tl, done } = timed(t, "text_ms", () => uprightTextLayer(page, page.render(s), s, mode));
    if (done) return textPage(pipeline, page, tl, t, dpi, imageOcr);
    // A layer the recognizer cannot read beats OCR, unless its codes are junk.
    const unreadable = tl.reason !== BROKEN_FONT_MAP && !readable(pipeline, tl.lines);
    if (tl.lines.length && (tl.trusted || mode === "text" || unreadable)) {
      return textPage(pipeline, page, tl, t, dpi, imageOcr);
    }
    note = tl.reason;
    // A broken font map still places its glyphs: when most run upright, the
    // page is upright (orient.onnx is not asked).
    turn = !(tl.reason === BROKEN_FONT_MAP && tl.stats.rotated < 0.5 * tl.stats.chars);
  }
  const gray = timed(t, "render_ms", () => page.render(pageScale(page, dpi)));
  const out = await pipeline.page(gray, turn);
  out.timings = { ...t, ...out.timings };
  out.timings["total_ms"] = Object.entries(out.timings).reduce(
    (n, [k, v]) => (k === "total_ms" ? n : n + v),
    0
  );
  out.note = `${note}; ${out.note}`;
  return out;
}

async function textPage(
  pipeline: IndraPipeline,
  page: PdfPageSource,
  tl: TextLayer,
  t: Record<string, number>,
  dpi: number,
  imageOcr: boolean
): Promise<Page> {
  const k = pageScale(page, dpi);
  const turn = tl.stats.turn ?? 0;
  const frame = tl.stats.frame ?? turn;
  const vertical = Boolean(tl.stats.vertical);
  const rendered = timed(t, "render_ms", () => page.render(k));
  const gray = frame ? rot90(rendered, frame) : rendered;
  let t0 = performance.now();
  const boxes = (await pipeline.detect(gray)).map(([x0, y0, x1, y1]): Box => [x0, y0, x1, y1]);
  t["det_ms"] = performance.now() - t0;
  const { toks, lost } = assignRuns(tl.lines, boxes, k);
  const rec: (RecLine | undefined)[] = toks.map((tk) => {
    const text = tk.length ? decodeToks(pipeline.vocab, tk) : "";
    return text ? { text, conf: 1, toks: tk } : undefined;
  });
  let empty = rec.flatMap((r, i) => (r ? [] : [i]));
  let source: Page["source"] = "text";
  // Sideways runs (a rotated table header, a chart's axis label) come from the
  // layer as their own lines; detected lines on them are not OCR'd.
  const side = tl.rotated.filter((r) => r.text.trim());
  const f = Math.fround;
  const sideBoxes = side.map((r): Box => [
    f(r.bbox[0] * k),
    f(r.bbox[1] * k),
    f(r.bbox[2] * k),
    f(r.bbox[3] * k),
  ]);
  if (side.length && empty.length) {
    empty = empty.filter((i) => !covered(item(boxes, i), sideBoxes));
  }
  if (empty.length && imageOcr && readable(pipeline, tl.lines)) {
    t0 = performance.now();
    const read = await pipeline.recognize(
      gray,
      empty.map((i) => item(boxes, i)),
      64,
      vertical
    );
    for (const [n, i] of empty.entries()) {
      const r = item(read, n);
      if (r.text.trim()) {
        rec[i] = r;
        source = "hybrid";
      }
    }
    t["rec_ms"] = performance.now() - t0;
  }
  if (pipeline.mathReader) {
    t0 = performance.now();
    await readMath(pipeline, gray, boxes, rec);
    t["math_ms"] = performance.now() - t0;
  }
  const keep = rec.flatMap((r, i) => (r ? [i] : []));
  const keptBoxes = keep.map((i) => item(boxes, i));
  const keptRec = keep.map((i) => item(rec, i) ?? { text: "", conf: 0, toks: [] });
  for (const [n, r] of side.entries()) {
    const b = item(sideBoxes, n);
    keptBoxes.push(b);
    const cx = f(b[0] + b[2]) / 2; // one x: a sideways line is never cut
    keptRec.push({ text: r.text, conf: 1, toks: [...r.text].map((c): Tok => [c, cx, r.seq]) });
  }
  const out = await pipeline.assemble(gray.width, gray.height, keptBoxes, keptRec, t, vertical);
  out.source = source;
  out.rotation = 90 * turn;
  if (vertical) {
    // Back on the page as it is: (x, y) -> (W - y, x).
    const w = gray.height;
    const back = ([x0, y0, x1, y1]: Box): Box => [w - y1, x0, w - y0, x1];
    out.lines = out.lines.map((ln) => ({ ...ln, box: back(ln.box) }));
    out.blocks = out.blocks.map((b) => ({ ...b, bbox: back(b.bbox) }));
    out.vertical = true;
  }
  if (lost) out.note += `; ${lost} chars outside detected lines`;
  return out;
}

/**
 * Text-layer lines that hold math (the classifier, on their crops) are read
 * by the math reader: the layer has the symbols but not the LaTeX. The line
 * keeps its place in the content order (its first character's index).
 */
async function readMath(
  pipeline: IndraPipeline,
  gray: Gray,
  boxes: readonly Box[],
  rec: (RecLine | undefined)[]
): Promise<void> {
  const math = pipeline.mathReader;
  if (!math) return;
  const idx = rec.flatMap((r, i) => {
    const first = r?.toks[0];
    return first && typeof first[0] === "string" && first.length > 2 ? [i] : [];
  });
  if (!idx.length) return;
  const crops = idx.map((i) => cropLine(gray, item(boxes, i), pipeline.height, pipeline.imageOps));
  const flagged = (await math.flags(crops)).flatMap((fl, n) => (fl ? [item(idx, n)] : []));
  if (!flagged.length) return;
  const readings = await math.read(
    gray,
    flagged.map((i) => item(boxes, i))
  );
  for (const [n, i] of flagged.entries()) {
    const { text, conf } = item(readings, n);
    const r = rec[i];
    if (!r || !text.includes("$") || conf < MATH_MIN_CONF) continue;
    const first = Math.min(...r.toks.flatMap((tk) => (tk[2] === undefined ? [] : [tk[2]])));
    const [x0, , x1] = item(boxes, i);
    rec[i] = { text, conf: 1, toks: [[text, Math.fround(x0 + x1) / 2, first]] };
  }
}

/** The bytes say PDF: "%PDF-" in the first 1024 bytes (the header may follow a little junk). */
export function isPdf(bytes: ArrayBuffer | Uint8Array): boolean {
  const view =
    bytes instanceof Uint8Array
      ? bytes.subarray(0, 1024)
      : new Uint8Array(bytes, 0, Math.min(1024, bytes.byteLength));
  for (let i = 0; i + 5 <= view.length; i++) {
    if (
      view[i] === 0x25 &&
      view[i + 1] === 0x50 &&
      view[i + 2] === 0x44 &&
      view[i + 3] === 0x46 &&
      view[i + 4] === 0x2d
    ) {
      return true;
    }
  }
  return false;
}
