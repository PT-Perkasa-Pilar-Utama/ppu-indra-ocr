import { readFileSync } from "fs";
import { join } from "path";

import type { Box } from "../../src/indra/rec-preproc.js";
import type { TableLine, WordText } from "../../src/indra/table-features.js";
import { GOLDEN_DIR } from "./golden.js";

/**
 * Fixtures from scripts/golden/tables.py (indra-ocr 1.2.0): the inputs and every
 * intermediate of table_rel.read_table and TableVerifier.apply, and crafted cases.
 */

/** A numpy array as base64 of little-endian bytes. */
export type Arr = { shape: number[]; b64: string };

export const f32 = (a: Arr): Float32Array => {
  const bytes = Buffer.from(a.b64, "base64");
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
};

export const i64 = (a: Arr): number[] => {
  const bytes = Buffer.from(a.b64, "base64");
  return Array.from(
    new BigInt64Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length)),
    Number
  );
};

export type StructJson = {
  cells: [number, number, number, number, number[]][];
  n_head: number;
  outside: number[];
  n_rows: number;
  n_cols: number;
};

export type WordJson = { box: number[]; text: string; line: number; pos: number; n: number };

/** One read_table call. */
export type ReadJson = {
  key: string;
  html: string | null;
  args: {
    lines: { box: number[]; toks: [number | string, number][]; space: boolean[] }[];
    mirror: boolean;
    col_spans: "geo" | "rel" | "rel2";
    split: boolean;
    refine: boolean;
  };
  decodes: { toks: [number | string, number][]; text: string }[];
  words: WordJson[];
  wf: { words: WordJson[]; bbox: number[]; feat: Arr };
  split_in: { shape: number[]; sha: string };
  maps: { bbox: number[]; line_h: number; result: { s: number; origin: number[] } | null };
  split_feat: Arr;
  norm_boxes: { boxes: Arr; out: Arr };
  word_ids: { n_vocab: number; ids: number[] };
  model_in: { n_feat: number };
  model_out: { pair: Arr; tag: Arr };
  tfeat: Arr;
  refine_out: { tag_out: Arr };
  decode: {
    tags: Arr;
    col_spans: "geo" | "rel" | "rel2";
    texts: string[];
    struct: StructJson | null;
  };
};

export type VariantJson = Pick<ReadJson, "args" | "decodes" | "html" | "decode"> & {
  label: string;
};

export type VerifyBlockJson = {
  key: string;
  html: string;
  n_tables: number;
  heads_ok: boolean;
  features: Record<string, number>;
  decided: boolean | null;
  rule: [number, number] | null;
  p_calls: [number, boolean][];
};

export type VerifyJson = {
  weights: "ocr" | "layer";
  gray: string | null;
  boxes: Arr;
  texts: string[];
  order: number[];
  cls: Arr;
  same: Arr;
  tab: Arr | null;
  cells: Record<string, [number[], string][]> | null;
  tables: Record<string, string> | null;
  out: { relabelled: boolean; cls: Arr; tables: Record<string, string> | null };
  blocks: VerifyBlockJson[];
  parts: Record<string, string[]>;
  label?: string;
};

export type PageJson = {
  tsr_rel_span: "geo" | "rel" | "rel2";
  tables: { gray: string; page_rtl: boolean; reads: ReadJson[]; tables: Record<string, string> }[];
  verify: VerifyJson[];
  variants?: VariantJson[];
};

export function readFixture<T>(name: string): T {
  // SAFETY: scripts/golden/tables.py writes each file in the shape its caller names.
  return JSON.parse(readFileSync(join(GOLDEN_DIR, "tables", name), "utf8")) as T;
}

/** The lines of a read as the table reader takes them. */
export function linesOf(args: ReadJson["args"]): TableLine[] {
  return args.lines.map((l) => ({
    box: l.box as unknown as Box,
    toks: l.toks.map(([id, x]) => [id, x] as const),
  }));
}

/**
 * The recognizer's answers, replayed: `isSpace` once per token in line order (as
 * `wordsOf` asks), `decode` by the tokens it is given.
 */
export function replay(args: ReadJson["args"], decodes: ReadJson["decodes"]): WordText {
  const spaces = args.lines.flatMap((l) => l.space);
  let next = 0;
  const texts = new Map(decodes.map((d) => [JSON.stringify(d.toks), d.text]));
  return {
    isSpace: () => {
      const v = spaces[next++];
      if (v === undefined) throw new RangeError("isSpace asked more often than the reference did");
      return v;
    },
    decode: (toks) => {
      const text = texts.get(JSON.stringify(toks.map((t) => [t[0], t[1]])));
      if (text === undefined) throw new RangeError("decode got tokens the reference never decoded");
      return text;
    },
  };
}

/** Largest absolute difference of two equally long arrays (Infinity when the lengths differ). */
export function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs((a[i] ?? 0) - (b[i] ?? 0)));
  return m;
}
