// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/table_rel.py (Apache-2.0, Copyright Datalab).
//
// Cells, rows and columns from the outputs of tsr_rel.onnx, and the HTML of the
// decoded table.
//
// `decodeTable` groups words into cells (average linkage on "same cell"), cells
// into rows and columns (average linkage on the mean pair logits), orders them by
// position, and gives a cell every row (column) whose words it overlaps
// vertically (horizontally) by more than half: a spanning cell.
//
// Ceiling against numpy: the double sums in `meanLogit` add in index order; numpy
// adds segments of 8 or more values pairwise. The results differ in the last bit
// of a float64 only.

import { meanF32 } from "./numeric.js";
import { cluster, medianF32 } from "./order-post.js";
import { LETTER, PY_SPACE, pyHtmlEscape } from "./py-text.js";
import type { Word } from "./table-features.js";
import { item } from "./util.js";

const f = Math.fround;

const SAME_CELL = 0;
const SAME_ROW = 1;
const SAME_COL = 2;
const TAG_HEAD = 0;
const TAG_OUT = 1;

/** How a cell's columns (and rows) are found: see `decodeTable`. */
export type ColSpans = "geo" | "rel" | "rel2";

/**
 * meta.json "tsr_rel_span" as a ColSpans: the reference takes a cell's spans
 * from the model's relations for "rel" and "rel2" and from geometry otherwise.
 */
export function colSpans(value: string | undefined): ColSpans {
  return value === "rel" || value === "rel2" ? value : "geo";
}

/** A cell: the rows r0..r1 and columns c0..c1 it spans (inclusive), and its words. */
export type TableCell = { r0: number; r1: number; c0: number; c1: number; words: number[] };

/** A decoded table. `outside`: words outside the table; `nHead`: header rows. */
export type TableStruct = {
  cells: TableCell[];
  nHead: number;
  outside: number[];
  nRows: number;
  nCols: number;
};

/**
 * `cluster` (average linkage, merge while the mean logit is positive) run on each
 * connected component of the positive pairs: two groups with no positive pair
 * between them never merge, so the result is the same, at a fraction of the cost
 * when positives are sparse (same cell). `L` is [n, n].
 */
export function clusterSparse(L: Float32Array, n: number): number[][] {
  const root = Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => {
    let r = i;
    while (item(root, r) !== r) r = item(root, r);
    return r;
  };
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (item(L, i * n + j) > 0) {
        const a = find(i);
        const b = find(j);
        root[Math.max(a, b)] = Math.min(a, b); // a component's root is its smallest member
      }
    }
  }
  const comps = new Map<number, number[]>();
  for (let i = 0; i < n; i++) comps.set(find(i), [...(comps.get(find(i)) ?? []), i]);
  const out: number[][] = [];
  for (const members of comps.values()) {
    if (members.length <= 2) {
      out.push(members); // one item, or two joined by a positive pair: one group
      continue;
    }
    const k = members.length;
    const sub = new Float32Array(k * k);
    for (const [a, i] of members.entries())
      for (const [b, j] of members.entries()) sub[a * k + b] = item(L, i * n + j);
    for (const g of cluster(sub, k)) out.push(g.map((x) => item(members, x)));
  }
  return out;
}

/**
 * Mean logit between every two groups of items: [g, g] float32 with a zero
 * diagonal. `L` is [n, n]; each group lists item indices.
 */
export function meanLogit(L: Float32Array, n: number, groups: readonly number[][]): Float32Array {
  const g = groups.length;
  // Sum the rows of each group first, then the columns, as numpy's reduceat does.
  const rowSums = groups.map((members) => {
    const s = new Float64Array(n);
    for (const i of members) for (let j = 0; j < n; j++) s[j] = (s[j] ?? 0) + item(L, i * n + j);
    return s;
  });
  const out = new Float32Array(g * g);
  for (let a = 0; a < g; a++) {
    for (let b = 0; b < g; b++) {
      if (a === b) continue;
      let sum = 0;
      for (const j of item(groups, b)) sum += item(item(rowSums, a), j);
      out[a * g + b] = sum / (item(groups, a).length * item(groups, b).length);
    }
  }
  return out;
}

/** A float32 value (`f32`) or a Python float, as numpy 2 types them in a mixed operation. */
type Scalar = { v: number; f32: boolean };

/**
 * The first and last of the bands the extent [lo, hi] covers over half of, always
 * including its own. `lo`, `hi` are float32 scalars, the bands Python floats, so
 * `min(hi, b) - max(lo, a)` is float32 when either side is, and its comparison
 * with `0.5 * (b - a)` rounds that float to float32.
 */
function span(
  lo: number,
  hi: number,
  bands: readonly (readonly [number, number])[],
  own: number
): [number, number] {
  const covered = new Set<number>([own]);
  for (const [k, [a, b]] of bands.entries()) {
    const top: Scalar = b < hi ? { v: b, f32: false } : { v: hi, f32: true };
    const bottom: Scalar = a > lo ? { v: a, f32: false } : { v: lo, f32: true };
    const rhs = 0.5 * (b - a);
    const wide =
      top.f32 || bottom.f32 ? f(f(top.v) - f(bottom.v)) > f(rhs) : top.v - bottom.v > rhs;
    if (wide) covered.add(k);
  }
  const sorted = [...covered].toSorted((p, q) => p - q);
  return [item(sorted, 0), item(sorted, sorted.length - 1)];
}

/** The run of groups (columns or rows) around `own` with a positive mean logit with one cell. */
function relSpan(row: readonly boolean[], own: number): [number, number] {
  let lo = own;
  let hi = own;
  while (lo > 0 && row[lo - 1]) lo--;
  while (hi + 1 < row.length && row[hi + 1]) hi++;
  return [lo, hi];
}

function groupOf(of: ReadonlyMap<number, number>, cell: number): number {
  const v = of.get(cell);
  if (v === undefined) throw new RangeError(`cell ${cell} is in no group`);
  return v;
}

/**
 * Cells, rows and columns from the model's outputs for N words. `pair` is
 * [3, N, N] logits, `tags` [N, 2] logits, `boxes` [N, 4] float32.
 *
 * `colSpans` "geo": a cell spans the columns whose band its words cover by more
 * than half; "rel": the run of columns around its own with which its words have a
 * positive mean "same column" logit (a narrow group header over the columns it
 * heads); "rel2": "rel" for columns, and the same for rows with the "same row"
 * logits (a stub header over the header rows it spans). `texts`: the words' texts.
 * Null when fewer than two words are inside the table.
 */
export function decodeTable(
  pair: Float32Array,
  tags: Float32Array,
  boxes: Float32Array,
  colSpans: ColSpans = "geo",
  texts?: readonly string[]
): TableStruct | null {
  const n = boxes.length / 4;
  if (n === 0) return null;
  if (pair.length !== 3 * n * n || tags.length !== 2 * n)
    throw new RangeError(
      `decodeTable: pair ${pair.length}, tags ${tags.length} do not fit ${n} words`
    );
  const plane = (c: number): Float32Array => pair.subarray(c * n * n, (c + 1) * n * n);
  const all = Array.from({ length: n }, (_, i) => i);
  const outside = all.filter((i) => item(tags, i * 2 + TAG_OUT) > 0);
  const keep = all.filter((i) => item(tags, i * 2 + TAG_OUT) <= 0);
  if (keep.length < 2) return null;
  const k = keep.length;
  const same = new Float32Array(k * k);
  const sameCell = plane(SAME_CELL);
  for (const [a, i] of keep.entries())
    for (const [b, j] of keep.entries()) same[a * k + b] = item(sameCell, i * n + j);
  const cells = clusterSparse(same, k).map((c) => c.map((x) => item(keep, x)));
  const m = cells.length;
  const bx = (i: number, j: number): number => item(boxes, i * 4 + j);
  const cb = cells.map((c) => [
    Math.min(...c.map((i) => bx(i, 0))),
    Math.min(...c.map((i) => bx(i, 1))),
    Math.max(...c.map((i) => bx(i, 2))),
    Math.max(...c.map((i) => bx(i, 3))),
  ]);
  const cbAt = (c: number, j: number): number => item(item(cb, c), j);
  const center = (g: readonly number[], lo: number, hi: number): number =>
    medianF32(g.map((c) => f(f(cbAt(c, lo) + cbAt(c, hi)) / 2)));
  const rows = clusterSparse(meanLogit(plane(SAME_ROW), n, cells), m).toSorted(
    (p, q) => center(p, 1, 3) - center(q, 1, 3)
  );
  const cols = clusterSparse(meanLogit(plane(SAME_COL), n, cells), m).toSorted(
    (p, q) => center(p, 0, 2) - center(q, 0, 2)
  );
  // Each row's (column's) band: the median extent of its cells.
  const bandsOf = (groups: number[][], lo: number, hi: number): [number, number][] =>
    groups.map((g): [number, number] => [
      medianF32(g.map((c) => cbAt(c, lo))),
      medianF32(g.map((c) => cbAt(c, hi))),
    ]);
  const rband = bandsOf(rows, 1, 3);
  const cband = bandsOf(cols, 0, 2);
  const rowOf = new Map(rows.flatMap((g, r) => g.map((c) => [c, r] as const)));
  const colOf = new Map(cols.flatMap((g, c) => g.map((x) => [x, c] as const)));
  /** [cells, groups]: true where a cell's mean logit with a group's cells is positive. */
  const positive = (ml: Float32Array, groups: readonly number[][]): boolean[][] =>
    Array.from({ length: m }, (_, c) =>
      groups.map((g) => meanF32(g.map((j) => item(ml, c * m + j))) > 0)
    );
  const relCols = colSpans === "rel" || colSpans === "rel2";
  const pc = relCols ? positive(meanLogit(plane(SAME_COL), n, cells), cols) : [];
  const pr = colSpans === "rel2" ? positive(meanLogit(plane(SAME_ROW), n, cells), rows) : [];
  const out: TableCell[] = cells.map((words, c) => {
    const [r0, r1] =
      colSpans === "rel2"
        ? relSpan(item(pr, c), groupOf(rowOf, c))
        : span(cbAt(c, 1), cbAt(c, 3), rband, groupOf(rowOf, c));
    const [c0, c1] = relCols
      ? relSpan(item(pc, c), groupOf(colOf, c))
      : span(cbAt(c, 0), cbAt(c, 2), cband, groupOf(colOf, c));
    return { r0, r1, c0, c1, words };
  });
  const head = new Float64Array(rows.length);
  for (const { r0, words } of out) {
    const mean = meanF32(words.map((i) => item(tags, i * 2 + TAG_HEAD)));
    head[r0] = Math.max(head[r0] ?? 0, mean);
  }
  let nHead = 0;
  while (nHead < rows.length - 1 && (head[nHead] ?? 0) > 0) nHead++;
  widenLoneCells(out, nHead, cols.length, texts);
  return { cells: out, nHead, outside, nRows: rows.length, nCols: cols.length };
}

/**
 * A body row that holds one cell with words, away from the first column (a unit
 * row such as "(In millions)" or a centered section title), spans the table's
 * width, as a section row does; a lone label in the first column stays a row
 * label, and a lone number stays in its column. Changes `cells` in place.
 */
function widenLoneCells(
  cells: TableCell[],
  nHead: number,
  nCols: number,
  texts?: readonly string[]
): void {
  const inRow = new Map<number, number[]>();
  for (const [k, { r0, r1 }] of cells.entries())
    for (let r = r0; r <= r1; r++) inRow.set(r, [...(inRow.get(r) ?? []), k]);
  for (const [r, ks] of inRow) {
    if (r < nHead || ks.length !== 1 || nCols < 3) continue;
    const k = item(ks, 0);
    const cell = item(cells, k);
    if (cell.r0 !== cell.r1 || cell.c0 === 0) continue;
    if (!texts || cell.words.some((i) => [...item(texts, i)].some((c) => LETTER.test(c))))
      cells[k] = { ...cell, c0: 0, c1: nCols - 1 };
  }
}

// Space-separated dot runs that end a cell.
const LEADER_DOTS = new RegExp(`(?:[${PY_SPACE}]+[.\\u00b7\\u2026]+)+$`, "u");

/** HTML for a decoded table; words outside it come before it as text. Null for a table of one row and one column. */
export function structHtml(struct: TableStruct, words: readonly Word[]): string | null {
  const R = struct.nRows;
  const C = struct.nCols;
  if (R < 2 && C < 2) return null;
  const occ = new Int32Array(R * C).fill(-1);
  const anchors = new Map<number, { r: number; c: number; rs: number; cs: number; ws: number[] }>();
  const taken = (r0: number, r1: number, c0: number, c1: number): boolean => {
    for (let r = r0; r <= Math.min(r1, R - 1); r++)
      for (let c = c0; c <= Math.min(c1, C - 1); c++) if (item(occ, r * C + c) >= 0) return true;
    return false;
  };
  for (const [k, cell] of struct.cells.entries()) {
    let { r1, c1 } = cell;
    const { r0, c0 } = cell;
    if (taken(r0, r1, c0, c1)) {
      // Overlapping spans: keep the cell to its own slot only.
      r1 = r0;
      c1 = c0;
      const k0 = item(occ, r0 * C + c0);
      if (k0 >= 0) {
        const anchor = anchors.get(k0);
        if (!anchor) throw new RangeError(`cell ${k0} has no anchor`);
        anchor.ws.push(...cell.words);
        continue;
      }
    }
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) occ[r * C + c] = k;
    anchors.set(k, { r: r0, c: c0, rs: r1 - r0 + 1, cs: c1 - c0 + 1, ws: [...cell.words] });
  }
  const text = (ws: readonly number[]): string => {
    const sorted = ws.toSorted(
      (a, b) =>
        item(words, a).line - item(words, b).line || item(words, a).box[0] - item(words, b).box[0]
    );
    const t = sorted.map((i) => item(words, i).text).join(" ");
    // leader dots between a label and its number are layout, not text
    return t.replace(LEADER_DOTS, "") || t;
  };
  const bySlot = new Map([...anchors.values()].map((a) => [a.r * C + a.c, a] as const));
  const out = ["<table>"];
  for (let r = 0; r < R; r++) {
    const tag = r < struct.nHead ? "th" : "td";
    let tds = "";
    for (let c = 0; c < C; c++) {
      const a = bySlot.get(r * C + c);
      if (a) {
        const attr =
          (a.rs > 1 ? ` rowspan="${a.rs}"` : "") + (a.cs > 1 ? ` colspan="${a.cs}"` : "");
        tds += `<${tag}${attr}>${pyHtmlEscape(text(a.ws))}</${tag}>`;
      } else if (item(occ, r * C + c) < 0) {
        tds += `<${tag}></${tag}>`;
      }
    }
    out.push(`<tr>${tds}</tr>`);
  }
  out.push("</table>");
  const pre = struct.outside.length ? text(struct.outside) : "";
  return (pre ? `${pre}\n\n` : "") + out.join("\n");
}
