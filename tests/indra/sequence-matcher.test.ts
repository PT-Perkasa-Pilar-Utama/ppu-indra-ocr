// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { SequenceMatcher } from "../../src/indra/sequence-matcher.js";
import { GOLDEN_DIR } from "./golden.js";

const same = (got: unknown, want: unknown): void => {
  expect(got).toEqual(want);
};

type MatcherCase = {
  a: string | string[];
  b: string | string[];
  opcodes: [string, number, number, number, number][];
  blocks: [number, number, number][];
  ratio: number;
  /** alo, ahi, blo, bhi, then the expected a, b, size */
  flm?: [number, number, number, number, number, number, number][];
};

// difflib.SequenceMatcher(None, a, b, autojunk=False) on CPython 3.12
// (scripts/golden/math-text.py).
const { cases }: { cases: MatcherCase[] } = JSON.parse(
  readFileSync(join(GOLDEN_DIR, "sequence-matcher.json"), "utf8")
);

describe("SequenceMatcher", () => {
  test("has fuzz cases for strings, string lists and long sequences", () => {
    expect(cases.length).toBeGreaterThan(300);
    expect(cases.some((c) => Array.isArray(c.a))).toBe(true);
    expect(cases.some((c) => c.a.length >= 250)).toBe(true);
  });

  test("opcodes, matching blocks and ratio equal CPython's on every case", () => {
    for (const c of cases) {
      const sm = new SequenceMatcher(c.a, c.b);
      const label = JSON.stringify([c.a, c.b]);
      same([label, sm.getOpcodes()], [label, c.opcodes]);
      same([label, sm.getMatchingBlocks().map((m) => [m.a, m.b, m.size])], [label, c.blocks]);
      expect([label, sm.ratio()]).toEqual([label, c.ratio]);
    }
  });

  test("findLongestMatch on a sub-range equals CPython's, ties included", () => {
    let n = 0;
    for (const c of cases) {
      const sm = new SequenceMatcher(c.a, c.b);
      for (const [alo, ahi, blo, bhi, a, b, size] of c.flm ?? []) {
        expect(sm.findLongestMatch(alo, ahi, blo, bhi)).toEqual({ a, b, size });
        n++;
      }
    }
    expect(n).toBeGreaterThan(300);
  });

  test("breaks ties on the earliest block in a, then in b", () => {
    // CPython docstring: " abcd" vs "abcd abcd" matches the first "abcd" in b
    expect(new SequenceMatcher(" abcd", "abcd abcd").findLongestMatch(0, 5, 0, 9)).toEqual({
      a: 0,
      b: 4,
      size: 5,
    });
    // "ab" vs "acab": the longest block is "ab" at (0, 2), not the "a" at (0, 0)
    expect(new SequenceMatcher("ab", "acab").getOpcodes()).toEqual([
      ["insert", 0, 0, 0, 2],
      ["equal", 0, 2, 2, 4],
    ]);
  });

  test("compares a string by code point and a list by element", () => {
    expect(new SequenceMatcher("a\u{1f600}b", "a\u{1f600}c").ratio()).toBeCloseTo(4 / 6, 15);
    expect(new SequenceMatcher(["ab", "c"], ["ab", "d"]).getOpcodes()).toEqual([
      ["equal", 0, 1, 0, 1],
      ["replace", 1, 2, 1, 2],
    ]);
  });

  test("two empty sequences have ratio 1 and one block", () => {
    const sm = new SequenceMatcher("", []);
    expect(sm.ratio()).toBe(1);
    expect(sm.getOpcodes()).toEqual([]);
    expect(sm.getMatchingBlocks()).toEqual([{ a: 0, b: 0, size: 0 }]);
  });
});
