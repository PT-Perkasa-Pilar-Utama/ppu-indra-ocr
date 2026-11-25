import { describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

import { forcedGaps, layerText, snippet, targetIds, writes } from "../../src/indra/agree.js";
import { layerFeatures, layerMatches } from "../../src/indra/layer-check.js";
import type { LayerRead } from "../../src/indra/layer-gaps.js";
import type { Tok } from "../../src/indra/types.js";
import { item } from "../../src/indra/util.js";
import { Vocab } from "../../src/indra/vocab.js";
import { GOLDEN_DIR, PRESET_CASES, events, field, loadCase, modelBytes } from "./golden.js";

type AgreeFixture = {
  gap_items: { blank: number[]; topv: number[][]; topi: number[][]; ids: number[][] }[];
  gaps: (number | "inf")[];
  targets: { vocab: string; text: string; visual: boolean; ids: number[][]; writes: boolean }[];
  tokens: [string, number, number][];
  snippets: { n: number; len: number; cut: number | null }[];
  layer_text: { rtl: boolean; text: string }[];
};

// The v1.2 meta.json: the local copy when there is one, else the cached download (CI).
const metaBytes = await modelBytes("meta", "v1.2");

// SAFETY: scripts/golden/agree.py writes this shape.
const fx = JSON.parse(readFileSync(join(GOLDEN_DIR, "agree.json"), "utf8")) as AgreeFixture;

describe("layer page score matches the reference", () => {
  const cases = PRESET_CASES.flatMap(([preset, name]) =>
    events(loadCase(name, preset), "layer_matches").map((e) => [`${preset} ${name}`, e] as const)
  );
  test.each(cases)("%s", (_, e) => {
    const reads = field<[number | null, number, string][]>(e, "reads").map(
      ([gap, conf, group]): LayerRead => ({ gap, conf, group })
    );
    const want = field<number[]>(e, "features");
    const got = layerFeatures(reads) ?? [];
    expect(got.length).toBe(want.length);
    for (const [i, v] of got.entries()) expect(v).toBeCloseTo(item(want, i), 5);
    expect(layerMatches(reads)).toBe(field<boolean>(e, "value"));
  });

  test("a page with no line that counts keeps its layer", () => {
    expect(layerFeatures([{ gap: null, conf: 0, group: "base" }])).toBeUndefined();
    expect(layerMatches([])).toBe(true);
  });
});

describe("forced CTC alignment matches the reference", () => {
  test("forcedGaps on one batch of random outputs", () => {
    const items = fx.gap_items.map((it) => ({
      blank: it.blank,
      topv: it.topv.flat(),
      topi: it.topi.flat(),
      k: it.topv[0]?.length ?? 8,
      ids: it.ids,
    }));
    const got = forcedGaps(items);
    for (const [i, want] of fx.gaps.entries()) {
      if (want === "inf") expect(item(got, i)).toBe(Infinity);
      else expect(item(got, i)).toBeCloseTo(want, 9);
    }
  });
});

describe("layer text helpers match the reference", () => {
  // SAFETY: meta.json ships with the models; its vocab fields are JSON strings.
  const meta = JSON.parse(new TextDecoder().decode(metaBytes)) as Record<string, string>;
  const vocabs = new Map([
    ["base", Vocab.fromJson(meta["vocab"] ?? "")],
    ["rtl", Vocab.fromJson(meta["vocab_rtl"] ?? "")],
    ["cjk", Vocab.fromJson(meta["vocab_cjk"] ?? "")],
  ]);
  const toks: Tok[] = fx.tokens.map(([c, x, i]) => [c, x, i]);

  test.each(fx.targets.map((t) => [t.text, t] as const))("targetIds and writes: %s", (_, t) => {
    const vocab = vocabs.get(t.vocab);
    if (!vocab) throw new Error(`no vocab ${t.vocab}`);
    expect(targetIds(t.text, vocab, t.visual).map((a) => [...a])).toEqual(t.ids);
    expect(writes(t.text, vocab)).toBe(t.writes);
  });

  test("snippet cuts at a word gap after n characters", () => {
    for (const s of fx.snippets) {
      const got = snippet(toks, s.n);
      expect(got.toks.length).toBe(s.len);
      expect(got.cut ?? null).toBe(s.cut);
    }
  });

  test("layerText orders words left to right, or right to left for an RTL recognizer", () => {
    for (const l of fx.layer_text) expect(layerText(toks, l.rtl)).toBe(l.text);
  });
});
