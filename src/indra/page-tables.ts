// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { Part } from "./assemble.js";
import { mirrorBox, partCells } from "./assemble.js";
import type { Gray } from "./image.js";
import type { TableMap } from "./layout-post.js";
import { TABLE, blockKey, splitBlocks } from "./layout-post.js";
import { decodeToks, isSpace } from "./lines.js";
import { isFormula, unspliced } from "./math-span-img.js";
import { blockClass } from "./order-post.js";
import type { WordText } from "./table-features.js";
import type { TableReader } from "./table-rel.js";
import type { VerifierSet } from "./table-verify.js";
import type { Lazy, ReadingOrder } from "./types.js";
import type { Vocab } from "./vocab.js";

/**
 * The table models of a model set: tsr_rel.onnx (with tsr_split and
 * tsr_refine) reads the cells of each Table block, and table_verify.json
 * relabels Table blocks it calls false tables (charts, prose). `layerTables`
 * false leaves PDF text-layer tables to the geometric assembly
 * (meta.json "tsr_rel_layer").
 */
export type TableModels = {
  reader?: Lazy<TableReader>;
  verifiers?: VerifierSet;
  layerTables: boolean;
};

/** The reading order with the verifier's classes, and the HTML read per table block. */
export type PageTables = { ro: ReadingOrder; tables?: TableMap; ms: number };

/**
 * Tables of a laid-out page. Lines with a spliced formula are read as the
 * recognizer read them: the formula is not a table feature. Text-layer pages
 * (`fromLayer`) get the verifier's own weight set, and no check when the
 * file has none (the OCR set rejected real tables there).
 */
export async function pageTables(
  models: TableModels,
  vocab: Vocab,
  gray: Gray,
  parts: readonly Part[],
  ro: ReadingOrder,
  pageRtl: boolean,
  fromLayer: boolean
): Promise<PageTables> {
  const { cls, same, order } = ro;
  if (!cls || !same) return { ro, ms: 0 };
  const t0 = performance.now();
  const n = parts.length;
  let tables: TableMap | undefined;
  if (models.reader && !(fromLayer && !models.layerTables)) {
    const reader = await models.reader();
    const text: WordText = {
      isSpace: (id) => isSpace(vocab, id),
      decode: (toks) => decodeToks(vocab, toks, pageRtl),
    };
    const read = new Map<string, string>();
    for (const blk of splitBlocks(order, same, n)) {
      // A block the model cannot read (fewer than two rows or columns) keeps the geometric table.
      if (blk.length < 2 || blockClass(cls, blk) !== TABLE) continue;
      const lines = blk.map((i) => {
        const p = parts[i];
        if (!p) throw new Error(`no line ${i}`);
        return { box: p.box, toks: unspliced(p.toks) };
      });
      const html = await reader.read(gray, lines, text, pageRtl);
      if (html) read.set(blockKey(blk), html);
    }
    tables = read;
  }
  const ms = performance.now() - t0;
  const verifier = fromLayer ? models.verifiers?.layer : models.verifiers?.ocr;
  if (!verifier) return { ro, tables, ms };
  const spliced = parts.map((p) => p.toks.some(isFormula));
  const res = verifier.apply({
    boxes: parts.map((p) => (pageRtl ? mirrorBox(p.box, gray.width) : p.box)),
    texts: parts.map((p, i) =>
      spliced[i] ? decodeToks(vocab, unspliced(p.toks), pageRtl) : p.text
    ),
    order,
    cls,
    same,
    cells: partCells(vocab, parts, pageRtl, gray.width, (p) => unspliced(p.toks)),
    tables,
    gray,
  });
  return res.relabelled
    ? { ro: { ...ro, cls: res.cls }, tables: res.tables, ms }
    : { ro, tables, ms };
}
