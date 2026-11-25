import { afterAll, beforeAll, describe, expect, test } from "bun:test";

import { ImageProcessor, cv } from "ppu-ocv";

import { opencvImageOps } from "../../src/core/opencv-image-ops.js";
import type { Gray, ImageOps } from "../../src/indra/image.js";
import { decodeTable, structHtml } from "../../src/indra/table-decode.js";
import type { Word } from "../../src/indra/table-features.js";
import { boxesF32, normBoxes, wordFeatures, wordsOf } from "../../src/indra/table-features.js";
import { splitFeatures, splitInput, splitMaps } from "../../src/indra/table-split.js";
import { refineTextFeatures, wordIds } from "../../src/indra/table-text.js";
import * as ort from "onnxruntime-node";
import { loadGray, releaseSessions, session, sha256 } from "./golden.js";
import type { PageJson, ReadJson, WordJson } from "./table-fixtures.js";
import { f32, linesOf, maxDiff, readFixture, replay } from "./table-fixtures.js";

/**
 * Every stage of table_rel.read_table on the pages that hold tables, against the
 * reference's intermediates: words, word features, the tsr_split input and
 * features, the model's text inputs, and the decoded table with its HTML.
 */

const CASES: string[] = ["table", "receipt", "sideways"];

const toWords = (ws: WordJson[]): Word[] =>
  ws.map((w) => ({ ...w, box: [w.box[0] ?? 0, w.box[1] ?? 0, w.box[2] ?? 0, w.box[3] ?? 0] }));

let ops: ImageOps;
let split: Awaited<ReturnType<typeof session>>;

beforeAll(async () => {
  await ImageProcessor.initRuntime();
  ops = opencvImageOps(cv);
  split = await session("tsrSplit");
});
afterAll(releaseSessions);

describe.each(CASES)("%s", (name) => {
  const page = readFixture<PageJson>(`${name}.json`);
  const reads = page.tables.flatMap((t) => t.reads.map((read) => ({ read, gray: t.gray })));

  test("the reference read at least one table", () => {
    expect(reads.length).toBeGreaterThan(0);
  });

  describe.each(reads)("table $read.key", ({ read, gray: file }) => {
    let gray: Gray;
    beforeAll(async () => {
      gray = await loadGray(file);
    });
    const words = toWords(read.words);
    const boxes = f32(read.norm_boxes.boxes);

    test("words: the same boxes and texts", () => {
      const got = wordsOf(linesOf(read.args), replay(read.args, read.decodes));
      expect(got).toEqual(words);
    });

    test("word features: bit-identical", () => {
      const bbox = read.wf.bbox as [number, number, number, number];
      const got = wordFeatures(toWords(read.wf.words), gray, bbox);
      expect(maxDiff(got, f32(read.wf.feat))).toBe(0);
    });

    test("tsr_split input: the same bytes", () => {
      const { bbox, line_h: lineH, result } = read.maps;
      const input = splitInput(ops, gray, bbox as [number, number, number, number], lineH);
      expect(input && [1, 1, input.ph, input.pw]).toEqual(read.split_in.shape);
      expect(input && sha256(input.x)).toBe(read.split_in.sha);
      expect([...(input?.origin ?? [])]).toEqual(result?.origin ?? []);
    });

    test("split features: within 1e-4", async () => {
      const { bbox, line_h: lineH } = read.maps;
      const maps = await splitMaps(
        split,
        ort.Tensor,
        ops,
        gray,
        bbox as [number, number, number, number],
        lineH
      );
      expect(maxDiff(splitFeatures(boxesF32(words), maps), f32(read.split_feat))).toBeLessThan(
        1e-4
      );
    });

    test("model inputs: box, word ids and text features", () => {
      expect(maxDiff(normBoxes(boxes), f32(read.norm_boxes.out))).toBe(0);
      const texts = words.map((w) => w.text);
      expect(wordIds(texts, read.word_ids.n_vocab)).toEqual(read.word_ids.ids);
      expect(maxDiff(refineTextFeatures(texts), f32(read.tfeat))).toBe(0);
    });

    test("decode: the same cells, rows and columns, then the same HTML", () => {
      const struct = decodeTable(
        f32(read.model_out.pair),
        f32(read.decode.tags),
        boxes,
        read.decode.col_spans,
        read.decode.texts
      );
      const want = read.decode.struct;
      expect(struct && toJson(struct)).toEqual(want);
      expect(struct && structHtml(struct, words)).toBe(read.html);
    });
  });
});

/** A decoded table in the reference's shape. */
function toJson(s: NonNullable<ReturnType<typeof decodeTable>>): ReadJson["decode"]["struct"] {
  return {
    cells: s.cells.map((c) => [c.r0, c.r1, c.c0, c.c1, c.words]),
    n_head: s.nHead,
    outside: s.outside,
    n_rows: s.nRows,
    n_cols: s.nCols,
  };
}
