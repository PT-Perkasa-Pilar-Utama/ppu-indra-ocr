// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/table_verify.py (Apache-2.0, Copyright Datalab).
//
// False-table verifier: a logistic model (table_verify.json in the models folder,
// optional) on features of every block the layout step classes Table. A block the
// model calls a false table (P >= the model's threshold) gets its best non-Table
// class instead, so it is written as text, a header, a list and so on.
//
// Features (`blockFeatures`): the table as written (rows, columns, cells, fill,
// words per cell, numeric cells, a two-column key-value pattern), the block text
// (math tokens, whole-formula lines, sentence ends, characters per line), the line
// geometry (height spread, rows by vertical centre, lines per row, gaps, distinct
// left / right edges, width spread), the block size on the page, the Table blocks
// on the page, ruling-line pixels in the block crop, the order model's class
// probabilities, whether the table heads were plausible and whether tsr_rel.onnx
// read the table.
//
// Ceilings against numpy: `pFalse` adds the 43 weighted features in index order,
// where numpy's `x @ w` calls BLAS (a different order of the float64 sum); `alignment`
// (table-block-features.ts) sorts line centres with a stable sort, numpy's argsort is not.

import type { Gray } from "./image.js";
import type { Cell, TableMap } from "./layout-post.js";
import { TABLE, blockKey, blockPart, splitBlocks } from "./layout-post.js";
import { sumF32 } from "./numeric.js";
import { blockClass, medianF32 } from "./order-post.js";
import { PY_SPACE, chars } from "./py-text.js";
import { pyLen, pySplit, pyStrip } from "./py-unicode.js";
import type { Box } from "./rec-preproc.js";
import type { BlockFeatures } from "./table-block-features.js";
import { alignment, parseTable, ruling, spreadF32 } from "./table-block-features.js";
import { headsPlausible, subTab } from "./table-heads.js";
import { item } from "./util.js";

const f = Math.fround;

const NUM = new RegExp(
  `^[${PY_SPACE}$${chars(0x20ac, 0xa3, 0xa5)}%()+\\-${chars(0x2013)}.,:/0-9]+$`
);
const MATH = new RegExp(
  `\\$|\\\\[A-Za-z]+|[=+${chars(0xd7, 0xf7, 0xb1, 0x2211, 0x222b, 0x221a, 0x221e, 0x2264, 0x2265, 0x2260, 0x2248)}^_{}]`,
  "g"
);
const WHOLE_MATH = /^\$[^$]+\$$/;
const SENT = new RegExp(`[.!?${chars(0x3002, 0xff01, 0xff1f)}]$`);

/** What `blockFeatures` reads about one Table block. */
export type BlockInput = {
  blk: readonly number[];
  boxes: readonly Box[];
  texts: readonly string[];
  /** Class logits [n, C] of the order model, flat. */
  cls: Float32Array;
  /** The block's HTML table, as written. */
  html: string;
  gray?: Gray;
  /** Table blocks on the page. */
  nTables: number;
  headsOk: boolean;
  /** Skip the ruling features (0): `apply` computes them only when they can change the decision. */
  withRuling?: boolean;
};

const mean = (v: readonly number[]): number => v.reduce((a, b) => a + b, 0) / v.length;

/** numpy's median of integers: the middle one, or the mean of the two middle ones. */
function medianOfInts(v: readonly number[]): number {
  const s = v.toSorted((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? item(s, m) : (item(s, m - 1) + item(s, m)) / 2;
}

/** Softmax of the block's summed class logits, in float32. */
function classProbs(cls: Float32Array, n: number, blk: readonly number[]): number[] {
  const c = cls.length / n;
  const z = Array.from({ length: c }, () => 0);
  for (const i of blk) for (let k = 0; k < c; k++) z[k] = f(item(z, k) + item(cls, i * c + k));
  const top = Math.max(...z);
  const p = z.map((v) => f(Math.exp(f(v - top))));
  const total = sumF32(p);
  return p.map((v) => f(v / total));
}

export function blockFeatures(input: BlockInput): BlockFeatures {
  const { blk, boxes, texts, cls, html, gray, nTables, headsOk, withRuling = true } = input;
  const b = blk.map((i) => item(boxes, i));
  const H = gray ? gray.height : Math.max(...boxes.map((x) => x[3]));
  const W = gray ? gray.width : Math.max(...boxes.map((x) => x[2]));
  const bb: Box = [
    Math.min(...b.map((x) => x[0])),
    Math.min(...b.map((x) => x[1])),
    Math.max(...b.map((x) => x[2])),
    Math.max(...b.map((x) => x[3])),
  ];
  const hts = b.map((x) => f(x[3] - x[1]));
  const pageH = medianF32(boxes.map((x) => f(x[3] - x[1]))) || 1;
  const rows = parseTable(html);
  const ncol = rows.length ? rows.map((r) => r.length) : [0];
  const maxc = Math.max(...ncol);
  const cells = rows.flat();
  const full = cells.filter(Boolean);
  const lt = blk.map((i) => pyStrip(item(texts, i)));
  const allTxt = lt.join(" ");
  const keyValue = (r: string[]): boolean =>
    r.length === 2 &&
    !!r[0] &&
    !!r[1] &&
    (r[0].endsWith(":") || (pyLen(r[0]) <= 40 && !NUM.test(r[0])));
  const kv = maxc === 2 && rows.length ? rows.filter(keyValue).length / rows.length : 0;
  const [rh, rv] = withRuling ? ruling(gray, bb) : [0, 0];
  const width = f(f(bb[2] - bb[0]) / f(Math.max(1, W)));
  const out: BlockFeatures = {
    lines: Math.log1p(blk.length),
    rows: Math.log1p(rows.length),
    cols: Math.log1p(medianOfInts(ncol)),
    maxcols: Math.log1p(maxc),
    cells: Math.log1p(cells.length),
    rows_le3: rows.length <= 3 ? 1 : 0,
    cols_2: maxc === 2 ? 1 : 0,
    fill: full.length / Math.max(1, rows.length * maxc),
    words_cell: Math.log1p(full.length ? mean(full.map((c) => pySplit(c).length)) : 0),
    num_cells: full.filter((c) => NUM.test(c)).length / Math.max(1, full.length),
    kv2: kv,
    math_tok: [...allTxt.matchAll(MATH)].length / Math.max(1, pySplit(allTxt).length),
    whole_math: lt.filter((t) => WHOLE_MATH.test(t)).length / Math.max(1, lt.length),
    sent_end: lt.filter((t) => SENT.test(t)).length / Math.max(1, lt.length),
    chars_line: Math.log1p(lt.length ? mean(lt.map(pyLen)) : 0),
    lh_cv: spreadF32(hts),
    lh_rel: f(medianF32(hts) / f(pageH)),
    area: f(f(f(bb[2] - bb[0]) * f(bb[3] - bb[1])) / f(Math.max(1, W * H))),
    width,
    tables_page: Math.log1p(nTables),
    rule_h: rh,
    rule_v: rv,
    heads_ok: headsOk ? 1 : 0,
  };
  for (const [k, p] of classProbs(cls, boxes.length, blk).entries()) out[`cls_${k}`] = p;
  return { ...out, ...alignment(b) };
}

/** One weight set: a logistic model over named features. */
export type VerifyWeights = {
  features: string[];
  mean: number[];
  scale: number[];
  weights: number[];
  bias: number;
  threshold: number;
};

/** table_verify.json: the weights for pages read by OCR and, optionally, for pages read from a PDF text layer. */
export type VerifyFile = VerifyWeights & { layer?: VerifyWeights };

/** What `TableVerifier.apply` reads about one page. */
export type VerifyInput = {
  boxes: readonly Box[];
  texts: readonly string[];
  order: readonly number[];
  /** Class logits [n, C], flat. */
  cls: Float32Array;
  /** Same-block logits [n, n], flat. */
  same: Float32Array;
  /** The order model's table heads [3, n, n], flat, when the page uses them. */
  tab?: Float32Array;
  cells?: ReadonlyMap<number, readonly Cell[]>;
  /** HTML read by tsr_rel.onnx per table block. */
  tables?: TableMap;
  gray?: Gray;
};

/** `cls` and `tables` as given when `relabelled` is false: the same objects. */
export type VerifyResult = { cls: Float32Array; tables: TableMap | undefined; relabelled: boolean };

/** The verifiers of a table_verify.json: for OCR pages, and for PDF text-layer pages (null when the file has none). */
export type VerifierSet = { ocr: TableVerifier; layer: TableVerifier | null };

const CLIP = 50; // a large score must not overflow exp

/** Python's np.nan_to_num for a float64. */
function finite(v: number): number {
  if (Number.isNaN(v)) return 0;
  return Math.min(Math.max(v, -Number.MAX_VALUE), Number.MAX_VALUE);
}

export class TableVerifier {
  public readonly threshold: number;

  public constructor(private readonly w: VerifyWeights) {
    const n = w.features.length;
    if (w.mean.length !== n || w.scale.length !== n || w.weights.length !== n)
      throw new RangeError(
        "table_verify weights: features, mean, scale and weights differ in length"
      );
    this.threshold = w.threshold;
  }

  /** The verifiers of a parsed table_verify.json. */
  public static load(file: VerifyFile): VerifierSet {
    return {
      ocr: new TableVerifier(file),
      layer: file.layer ? new TableVerifier(file.layer) : null,
    };
  }

  /**
   * The decision (P(false) >= threshold) when it holds for every value of the
   * ruling features (each in [0, 1]), else null: `ruling` then has to run. The same
   * decisions as computing everything.
   */
  public decided(feats: BlockFeatures): boolean | null {
    const decisions = new Set<boolean>();
    for (const rule_h of [0, 1])
      for (const rule_v of [0, 1])
        decisions.add(this.pFalse({ ...feats, rule_h, rule_v }) >= this.threshold);
    return decisions.size === 1 ? (decisions.values().next().value ?? null) : null;
  }

  public pFalse(feats: BlockFeatures): number {
    const { features, mean: mu, scale, weights, bias } = this.w;
    let dot = 0;
    for (const [k, name] of features.entries()) {
      const v = feats[name];
      if (v === undefined) throw new RangeError(`table_verify: no feature "${name}"`);
      dot += ((finite(v) - item(mu, k)) / item(scale, k)) * item(weights, k);
    }
    const z = Math.min(Math.max(dot + bias, -CLIP), CLIP);
    return 1 / (1 + Math.exp(-z));
  }

  /**
   * Every false table relabelled: its lines lose the Table class and its tsr_rel
   * table is dropped. The inputs are returned unchanged when no block is relabelled.
   */
  public apply(input: VerifyInput): VerifyResult {
    const { boxes, texts, order, cls, same, tab, cells, tables, gray } = input;
    const n = boxes.length;
    const c = cls.length / n;
    const blocks = splitBlocks(order, same, n).filter((blk) => blockClass(cls, blk) === TABLE);
    let outCls = cls;
    let dropped: Map<string, string> | undefined; // a copy of `tables`, made on the first relabelled block
    for (const blk of blocks) {
      const key = blockKey(blk);
      const read = tables?.get(key);
      const html = read ?? blockPart(blk, boxes, texts, cls, true, cells, tab) ?? "";
      const lineBoxes = blk.map((i) => item(boxes, i));
      const headsOk = blk.length > 1 && !!tab && headsPlausible(lineBoxes, subTab(tab, n, blk));
      const feats = blockFeatures({
        blk,
        boxes,
        texts,
        cls,
        html,
        gray,
        nTables: blocks.length,
        headsOk,
        withRuling: false,
      });
      feats["tsr"] = read === undefined ? 0 : 1;
      // The ruling features cost most of the time: compute them only when they can change the decision
      // (the score is linear in each, so its extremes are at the corners of [0, 1] x [0, 1]).
      let isFalse = this.decided(feats);
      if (isFalse === null) {
        const bb: Box = [
          Math.min(...lineBoxes.map((x) => x[0])),
          Math.min(...lineBoxes.map((x) => x[1])),
          Math.max(...lineBoxes.map((x) => x[2])),
          Math.max(...lineBoxes.map((x) => x[3])),
        ];
        [feats["rule_h"], feats["rule_v"]] = ruling(gray, bb);
        isFalse = this.pFalse(feats) >= this.threshold;
      }
      if (!isFalse) continue;
      if (outCls === cls) {
        outCls = cls.slice();
        if (tables?.size) dropped = new Map(tables);
      }
      for (const i of blk) outCls[i * c + TABLE] = -1e4;
      dropped?.delete(key);
    }
    return { cls: outCls, tables: dropped ?? tables, relabelled: outCls !== cls };
  }
}
