import { afterAll, describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

import { decodeToks } from "../../src/indra/lines.js";
import { splice, unspliced } from "../../src/indra/math-span-img.js";
import type { IndraPipeline } from "../../src/indra/pipeline.js";
import type { Box } from "../../src/indra/rec-preproc.js";
import { symbolCandidates, symbolWords } from "../../src/indra/symbol-digits.js";
import { readTurn } from "../../src/indra/turn-check.js";
import type { Tok } from "../../src/indra/types.js";
import { estimateSkew, preview } from "../../src/indra/upright.js";
import { GOLDEN_DIR, loadGray, makePipeline, releaseSessions } from "./golden.js";

/**
 * Code paths of indra-ocr 1.2.0 that the page goldens do not reach, against
 * the reference's own outputs (scripts/golden/paths.py).
 */

type TokJson = [string, number, "formula" | "pad" | null];
type Paths = {
  read_turn: { page: string; probs: string; p: number[]; skew: number; k: number }[];
  accept_in_table: {
    text: string;
    conf: number;
    draft: string;
    rconf: number;
    out: string | null;
  }[];
  splices: {
    label: string;
    toks: [string, number][];
    x0: number;
    x1: number;
    latex: string;
    page_rtl: boolean;
    spliced: TokJson[];
    unspliced: TokJson[];
    decoded: string;
    decoded_plain: string;
  }[];
  symbol_words: {
    line: [string, number, number][];
    box: number[];
    sym: number[][];
    words: [number, number, number[]][];
  };
  symbol_candidates: { layer: string; read: string; hits: number[]; out: Record<string, string> }[];
};

// SAFETY: scripts/golden/paths.py writes this shape.
const fx = JSON.parse(readFileSync(join(GOLDEN_DIR, "paths.json"), "utf8")) as Paths;
const toBox = ([x0 = 0, y0 = 0, x1 = 0, y1 = 0]: readonly number[]): Box => [x0, y0, x1, y1];
const tokJson = (t: Tok): TokJson => [
  String(t[0]),
  Math.round(t[1] * 1e4) / 1e4,
  t[3]?.kind ?? null,
];

let pipeline: IndraPipeline | undefined;
async function v12(): Promise<IndraPipeline> {
  pipeline ??= await makePipeline(true, "opencv", "v1.2");
  return pipeline;
}

describe("readTurn picks the reference's turn", () => {
  test.each(fx.read_turn.map((t) => [`${t.page} ${t.probs}`, t] as const))(
    "%s",
    async (_, t) => {
      const p = await v12();
      const gray = await loadGray(t.page);
      const small = preview(gray, p.imageOps);
      const skew = estimateSkew(small, p.imageOps);
      expect(skew).toBeCloseTo(t.skew, 6);
      expect(await readTurn(p, gray, skew, small, t.p)).toBe(t.k);
    },
    60_000
  );
});

describe("the accept head looks again at table lines", () => {
  test("recorded readings as table lines (in_table 1)", async () => {
    const math = (await v12()).mathReader;
    if (!math) throw new Error("no math reader");
    for (const a of fx.accept_in_table) {
      expect(math.accept(a.text, a.conf, a.draft, a.rconf, 1.0)).toBe(a.out);
    }
  });
});

describe("inline formula splices match the reference", () => {
  test.each(fx.splices.map((s) => [s.label, s] as const))("%s", async (_, s) => {
    const vocab = (await v12()).vocab;
    const toks = s.toks.map(([c, x]): Tok => [c, x]);
    const sp = splice(toks, s.x0, s.x1, s.latex);
    expect(sp.map(tokJson)).toEqual(s.spliced);
    expect(unspliced(sp).map(tokJson)).toEqual(s.unspliced);
    expect(decodeToks(vocab, sp, s.page_rtl)).toBe(s.decoded);
    expect(decodeToks(vocab, unspliced(sp), s.page_rtl)).toBe(s.decoded_plain);
  });
});

describe("symbol-digit helpers match the reference", () => {
  test("symbolWords", () => {
    const w = fx.symbol_words;
    const toks = w.line.map(([c, x, i]): Tok => [c, x, i]);
    const got = symbolWords(toks, toBox(w.box), w.sym.map(toBox));
    expect(got.map((s) => [s.start, s.end, s.hits])).toEqual(w.words);
  });

  test.each(fx.symbol_candidates.map((c) => [`${c.layer} / ${c.read}`, c] as const))(
    "symbolCandidates %s",
    (_, c) => {
      const got = Object.fromEntries(
        [...symbolCandidates(c.layer, c.read, c.hits)].map(([k, v]) => [String(k), v])
      );
      expect(got).toEqual(c.out);
    }
  );
});

afterAll(releaseSessions);
