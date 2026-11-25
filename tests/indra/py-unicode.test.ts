// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { fold, look } from "../../src/indra/math-text.js";
import {
  greekSmallLetter,
  hasLatinName,
  isAlpha,
  isCategoryM,
  isCombining,
  isDigit,
  isUpper,
  pySplit,
  pyStrip,
} from "../../src/indra/py-unicode.js";
import { isPySpace } from "../../src/indra/util.js";
import { GOLDEN_DIR } from "./golden.js";

const same = (got: unknown, want: unknown): void => {
  expect(got).toEqual(want);
};

type Fixture = {
  char_columns: string[];
  chars: (string | number)[][];
  assigned_ranges: [number, number][];
  char_hashes: Record<string, string | number>;
  greek: Record<string, string | null>;
};

// Python 3.12 (Unicode 15.0.0): scripts/golden/math-text.py.
const fx: Fixture = JSON.parse(readFileSync(join(GOLDEN_DIR, "math-text.json"), "utf8"));

describe("Python's character rules", () => {
  const col = (name: string): number => {
    const i = fx.char_columns.indexOf(name);
    if (i < 0) throw new Error(`no column ${name}`);
    return i + 1;
  };
  const cell = (row: (string | number)[], name: string): string | number => {
    const v = row[col(name)];
    if (v === undefined) throw new Error(`no cell ${name}`);
    return v;
  };

  test("the sample characters have Python's properties", () => {
    expect(fx.chars.length).toBeGreaterThan(1000);
    for (const row of fx.chars) {
      const c = String(row[0]);
      const at = c.codePointAt(0)?.toString(16);
      const got = {
        alpha: Number(isAlpha(c)),
        upper: Number(isUpper(c)),
        digit: Number(isDigit(c)),
        space: Number(isPySpace(c)),
        lower: c.toLowerCase(),
        fold: fold(c),
        look: look(c),
        combining: Number(isCombining(c)),
        cat_m: Number(isCategoryM(c)),
        latin_name: Number(hasLatinName(c)),
        nfkd: c.normalize("NFKD"),
        nfkc: c.normalize("NFKC"),
      };
      const want = Object.fromEntries(Object.keys(got).map((k) => [k, cell(row, k)]));
      same([at, got], [at, want]);
    }
  });

  // Every code point that Unicode 15.0 assigns, one property at a time. A failure
  // names the property: run scripts/golden/math-text.py on a Python of that
  // Unicode version to see the first code point that differs. Code points added
  // after 15.0 follow the JS engine and are not compared.
  const hex = (s: string): string =>
    Array.from(s, (c) => (c.codePointAt(0) ?? 0).toString(16)).join(".");
  const ALNUM = /^[\p{L}\p{N}]$/u;
  const props = {
    alpha: (c: string) => Number(isAlpha(c)),
    upper: (c: string) => Number(isUpper(c)),
    digit: (c: string) => Number(isDigit(c)),
    space: (c: string) => Number(isPySpace(c)),
    alnum: (c: string) => Number(ALNUM.test(c)),
    lower: (c: string) => hex(c.toLowerCase()),
    fold: (c: string) => hex(fold(c)),
    look: (c: string) => hex(look(c)),
    combining: (c: string) => Number(isCombining(c)),
    cat_m: (c: string) => Number(isCategoryM(c)),
    latin_name: (c: string) => Number(hasLatinName(c)),
    nfkd: (c: string) => hex(c.normalize("NFKD")),
    nfkc: (c: string) => hex(c.normalize("NFKC")),
  };
  const digests = Object.fromEntries(Object.keys(props).map((k) => [k, createHash("sha256")]));
  const pending = Object.fromEntries(Object.keys(props).map((k) => [k, [] as string[]]));
  const flush = (k: string): void => {
    digests[k]?.update(pending[k]?.join("") ?? "");
    if (pending[k]) pending[k].length = 0;
  };
  for (const [lo, hi] of fx.assigned_ranges) {
    for (let cp = lo; cp <= hi; cp++) {
      const c = String.fromCodePoint(cp);
      for (const [k, f] of Object.entries(props)) {
        const lines = pending[k];
        lines?.push(`${cp.toString(16)}:${f(c)}\n`);
        if (lines && lines.length >= 8192) flush(k);
      }
    }
  }
  for (const k of Object.keys(props)) flush(k);
  for (const k of Object.keys(props)) {
    test(`${k} matches Python over every assigned code point`, () => {
      same([k, digests[k]?.copy().digest("hex")], [k, fx.char_hashes[k]]);
    });
  }

  test("greekSmallLetter resolves what unicodedata.lookup resolves", () => {
    for (const [name, letter] of Object.entries(fx.greek)) {
      same([name, greekSmallLetter(name) ?? null], [name, letter]);
    }
    expect(greekSmallLetter("lambda")).toBeUndefined();
    expect(greekSmallLetter("lamda")).toBe(String.fromCharCode(0x3bb));
  });

  test("pySplit and pyStrip use Python's whitespace", () => {
    const s = (...cps: number[]): string => String.fromCharCode(...cps);
    const text = `a${s(0x1c)}b${s(0x85)}c${s(0xa0)}d${s(0xfeff)}e  f`;
    expect(pySplit(text)).toEqual(["a", "b", "c", `d${s(0xfeff)}e`, "f"]);
    expect(pyStrip(`${s(0x3000)} x ${s(0x1f)}`)).toBe("x");
    expect(pyStrip(s(0xfeff))).toBe(s(0xfeff));
  });
});
