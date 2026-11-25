import { describe, expect, test } from "bun:test";

import { crc32 } from "../../src/indra/table-text.js";
import { pyHtmlEscape, pyHtmlUnescape } from "../../src/indra/py-text.js";
import { hasLatinName, isAlpha, isDigit } from "../../src/indra/py-unicode.js";
import type { ColSpans, TableStruct } from "../../src/indra/table-decode.js";
import { decodeTable, structHtml } from "../../src/indra/table-decode.js";
import type { Word } from "../../src/indra/table-features.js";
import { refineTextFeatures, wordIds } from "../../src/indra/table-text.js";
import type { Arr, StructJson, WordJson } from "./table-fixtures.js";
import { f32, maxDiff, readFixture } from "./table-fixtures.js";

/**
 * The pure functions of table_rel.py on crafted inputs, against the reference's
 * outputs (scripts/golden/tables.py): decode and table_html with spans, outside
 * words, leader dots and markup characters; word ids; refiner text features.
 */

type Units = {
  decode: {
    pair: Arr;
    tags: Arr;
    boxes: Arr;
    col_spans: ColSpans;
    texts: string[] | null;
    struct: StructJson | null;
  }[];
  table_html: { words: WordJson[]; cases: { struct: StructJson; html: string | null }[] };
  word_ids: {
    n_vocab: number;
    texts: string[];
    ids: number[];
    crc32: { s: string; crc: number }[];
  };
  latin_ranges: [number, number][];
  digit_ranges: [number, number][];
  refine_text_features: { texts: string[]; out: Arr };
  html_unescape: { s: string; out: string }[];
};

const units = readFixture<Units>("units.json");

const fromJson = (s: StructJson): TableStruct => ({
  cells: s.cells.map(([r0, r1, c0, c1, words]) => ({ r0, r1, c0, c1, words })),
  nHead: s.n_head,
  outside: s.outside,
  nRows: s.n_rows,
  nCols: s.n_cols,
});

const toJson = (s: TableStruct): StructJson => ({
  cells: s.cells.map((c) => [c.r0, c.r1, c.c0, c.c1, c.words]),
  n_head: s.nHead,
  outside: s.outside,
  n_rows: s.nRows,
  n_cols: s.nCols,
});

describe("decodeTable", () => {
  test.each(units.decode.map((c, i) => [i, c.col_spans, c] as const))(
    "case %d (%s) groups the same cells, rows and columns",
    (_, __, c) => {
      const got = decodeTable(
        f32(c.pair),
        f32(c.tags),
        f32(c.boxes),
        c.col_spans,
        c.texts ?? undefined
      );
      expect(got && toJson(got)).toEqual(c.struct);
    }
  );

  test("the cases reach a null table, a stub header and an outside word", () => {
    expect(units.decode.some((c) => c.struct === null)).toBe(true);
    expect(units.decode.some((c) => (c.struct?.outside.length ?? 0) > 0)).toBe(true);
    expect(units.decode.some((c) => c.struct?.cells.some((x) => x[1] > x[0]))).toBe(true);
  });

  test("tensors that do not fit the words are refused", () => {
    const c = units.decode[0];
    if (!c) throw new Error("no decode case");
    expect(() => decodeTable(f32(c.pair).slice(1), f32(c.tags), f32(c.boxes))).toThrow(RangeError);
  });
});

describe("structHtml", () => {
  const words: Word[] = units.table_html.words.map((w) => ({
    ...w,
    box: [w.box[0] ?? 0, w.box[1] ?? 0, w.box[2] ?? 0, w.box[3] ?? 0],
  }));
  test.each(units.table_html.cases.map((c, i) => [i, c] as const))("case %d", (_, c) => {
    expect(structHtml(fromJson(c.struct), words)).toBe(c.html);
  });
});

describe("word ids", () => {
  test("the same hashed bucket for each text, 0 for a letter outside the Latin script", () => {
    const { texts, ids, n_vocab: nVocab } = units.word_ids;
    expect(wordIds(texts, nVocab)).toEqual(ids);
    expect(ids).toContain(0);
    expect(ids.filter((i) => i > 0).length).toBeGreaterThan(8);
  });

  test("crc32 matches zlib", () => {
    for (const { s, crc } of units.word_ids.crc32)
      expect(crc32(new TextEncoder().encode(s))).toBe(crc);
  });
});

describe("Python str.isdigit and Latin-named letters", () => {
  // The edges of every range, and one code point past each side: a missing or extra range fails.
  const edges = (r: [number, number][]): number[] => r.flatMap(([a, b]) => [a - 1, a, b, b + 1]);
  const inside = (r: [number, number][], cp: number): boolean =>
    r.some(([a, b]) => a <= cp && cp <= b);

  test("isDigit", () => {
    for (const cp of edges(units.digit_ranges))
      expect(isDigit(String.fromCodePoint(cp))).toBe(inside(units.digit_ranges, cp));
  });

  test("letters named LATIN", () => {
    for (const cp of edges(units.latin_ranges)) {
      const c = String.fromCodePoint(cp);
      expect(isAlpha(c) && hasLatinName(c)).toBe(inside(units.latin_ranges, cp));
    }
  });
});

describe("refineTextFeatures", () => {
  test("the same 13 features for numbers, dashes, marks, letters and currency", () => {
    const { texts, out } = units.refine_text_features;
    expect(maxDiff(refineTextFeatures(texts), f32(out))).toBe(0);
  });
});

describe("pyHtmlEscape", () => {
  test("escapes & < > and leaves quotes", () => {
    expect(pyHtmlEscape(`a & b < c > "d" 'e'`)).toBe(`a &amp; b &lt; c &gt; "d" 'e'`);
  });
});

describe("pyHtmlUnescape", () => {
  test.each(units.html_unescape.map((c, i) => [i, c] as const))(
    "case %d is html.unescape's",
    (_, c) => {
      expect(pyHtmlUnescape(c.s)).toBe(c.out);
    }
  );
});
