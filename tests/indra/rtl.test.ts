// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { describe, expect, test } from "bun:test";
import { join } from "node:path";

import {
  bracketsAsShapes,
  clusters,
  flip,
  hasRtl,
  labelFromVisual,
  normalize,
  rtlBase,
  visualLabel,
} from "../../src/indra/rtl.js";
import {
  bidiClass,
  displayLen,
  eastAsianWide,
  DIGIT,
  LTR,
  MARK,
  RTL,
} from "../../src/indra/unicode.js";
import { GOLDEN_DIR } from "./golden.js";

const s = (...cps: number[]): string => String.fromCodePoint(...cps);
const AR = s(0x645, 0x631, 0x62d, 0x628, 0x627);
const HE = s(0x5e9, 0x5dc, 0x5d5, 0x5dd);
const rev = (t: string): string => [...t].reverse().join("");

/** Seeded PRNG (mulberry32), so the random test is the same on every run. */
function prng(seed: number): () => number {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("flip", () => {
  test("reverses RTL text and keeps LTR runs", () => {
    expect(flip(AR)).toBe(rev(AR));
    const v = flip(`${AR} 2026-10-13 ${HE} abc def`);
    expect(v).toContain("abc def");
    expect(v).toContain("2026-10-13");
    expect(v.indexOf("abc def")).toBeLessThan(v.indexOf(rev(HE)));
    expect(v.indexOf(rev(HE))).toBeLessThan(v.indexOf("2026-10-13"));
    expect(flip(`the word ${HE} means peace`)).toBe(`the word ${rev(HE)} means peace`);
  });

  test("is its own inverse on random strings", () => {
    const rnd = prng(0);
    const alphabet = [...`${AR}${HE}abc 019()[].,:-`, s(0x64e), s(0x661)];
    for (let n = 0; n < 1000; n++) {
      const len = 1 + Math.floor(rnd() * 29);
      let t = "";
      for (let i = 0; i < len; i++) t += alphabet[Math.floor(rnd() * alphabet.length)];
      const u = normalize(t);
      expect(flip(flip(u))).toBe(u);
    }
  });

  test("returns text without RTL letters as is", () => {
    expect(flip("abc (def) 123")).toBe("abc (def) 123");
  });

  test("mirrors brackets at RTL level only when asked", () => {
    const t = `${HE}(${AR})`;
    expect(flip(t)).toBe(`(${rev(AR)})${rev(HE)}`);
    expect(flip(t, false)).toBe(`)${rev(AR)}(${rev(HE)}`);
  });

  test("forces the paragraph direction with baseRtl", () => {
    const t = `abc ${HE} def`;
    expect(flip(t, true, null)).toBe(flip(t));
    expect(flip(t, true, true)).toBe(`def ${rev(HE)} abc`);
    expect(flip(t, true, false)).toBe(`abc ${rev(HE)} def`);
  });
});

describe("normalize and clusters", () => {
  test("maps presentation forms to base letters and drops tatweel", () => {
    expect(normalize(s(0xfe8d, 0x640, 0x640, 0x645))).toBe(s(0x627, 0x645));
    expect(normalize(s(0xfb2a))).toBe(s(0x5e9, 0x5c1));
  });

  test("drops a combining mark without a base", () => {
    expect(normalize(s(0x64e, 0x645))).toBe(s(0x645));
    expect(normalize(s(0x645, 0x64e))).toBe(s(0x645, 0x64e));
  });

  test("keeps marks with their base", () => {
    expect(clusters(s(0x645, 0x64e, 0x631, 0x1f600))).toEqual([
      s(0x645, 0x64e),
      s(0x631),
      s(0x1f600),
    ]);
  });

  test("labels are flipped normalized text", () => {
    const t = s(0xfe8d, 0x640, 0x645);
    expect(visualLabel(t)).toBe(flip(normalize(t)));
    expect(labelFromVisual(flip(t))).toBe(visualLabel(t));
  });
});

describe("rtlBase and hasRtl", () => {
  test("counts RTL letters against LTR letters", () => {
    expect(hasRtl("abc")).toBe(false);
    expect(hasRtl(`abc${HE}`)).toBe(true);
    expect(rtlBase(`${HE} ab`)).toBe(true);
    expect(rtlBase(`${HE} abcde`)).toBe(false);
    expect(rtlBase("")).toBe(false);
  });
});

describe("bracketsAsShapes", () => {
  test("votes by the first bracket of RTL lines without Latin letters", () => {
    expect(bracketsAsShapes([`${HE}(${AR}`, `${AR}(${HE}`])).toBe(true);
    expect(bracketsAsShapes([`${HE})${AR}`, `${AR}(${HE}`])).toBe(false);
    expect(bracketsAsShapes([`${HE}(a${AR}`])).toBe(false);
    expect(bracketsAsShapes([])).toBe(false);
  });
});

describe("unicode lookups", () => {
  test("classes and widths of known characters", () => {
    expect(bidiClass(0x5d0)).toBe(RTL);
    expect(bidiClass(0x627)).toBe(RTL);
    expect(bidiClass(0x61)).toBe(LTR);
    expect(bidiClass(0x31)).toBe(DIGIT);
    expect(bidiClass(0x661)).toBe(DIGIT);
    expect(bidiClass(0x64e)).toBe(MARK);
    expect(bidiClass(0x20)).toBe(0);
    expect(eastAsianWide(0x4e2d)).toBe(true);
    expect(eastAsianWide(0xff21)).toBe(true);
    expect(eastAsianWide(0x41)).toBe(false);
  });

  test("displayLen counts wide characters twice", () => {
    expect(displayLen(s(0x4e2d, 0x6587, 0x61))).toBe(5);
    expect(displayLen(s(0x1f600))).toBe(2);
    expect(displayLen("")).toBe(0);
  });
});

type Sample = {
  unicode: string;
  chars: [number, string, string, string][];
  strings: {
    s: string;
    flip: string;
    flipNoMirror: string;
    flipRtl: string;
    flipLtr: string;
    normalize: string;
    rtlBase: boolean;
    visualLabel: string;
    visualLabelRtl: string;
    labelFromVisual: string;
    labelFromVisualShapes: string;
  }[];
  brackets: { lines: string[]; shapes: boolean }[];
};

function classOf(bidi: string, cat: string): number {
  if (bidi === "R" || bidi === "AL") return RTL;
  if (bidi === "L") return LTR;
  if (bidi === "EN" || bidi === "AN") return DIGIT;
  return cat === "Mn" || cat === "Me" ? MARK : 0;
}

const sample: Sample = await Bun.file(join(GOLDEN_DIR, "unicode-sample.json")).json();

describe("parity with the Python reference (Unicode 15.0.0)", () => {
  test("bidi class, East Asian width and marks match for sampled code points", () => {
    const bad: unknown[] = [];
    for (const [cp, bidi, eaw, cat] of sample.chars) {
      const k = bidiClass(cp);
      const wantK = classOf(bidi, cat);
      if (k !== wantK || eastAsianWide(cp) !== (eaw === "W" || eaw === "F"))
        bad.push([cp, k, bidi, eaw]);
    }
    expect(bad).toEqual([]);
  });

  test("flip, normalize, labels and rtlBase match for sample strings", () => {
    for (const r of sample.strings) {
      const got = {
        flip: flip(r.s),
        flipNoMirror: flip(r.s, false),
        flipRtl: flip(r.s, true, true),
        flipLtr: flip(r.s, true, false),
        normalize: normalize(r.s),
        rtlBase: rtlBase(r.s),
        visualLabel: visualLabel(r.s),
        visualLabelRtl: visualLabel(r.s, true),
        labelFromVisual: labelFromVisual(r.s),
        labelFromVisualShapes: labelFromVisual(r.s, true),
      };
      const { s: _s, ...want } = r;
      expect({ s: r.s, ...got }).toEqual({ s: r.s, ...want });
    }
  });

  test("bracketsAsShapes matches for sample line lists", () => {
    for (const b of sample.brackets)
      expect([b.lines, bracketsAsShapes(b.lines)]).toEqual([b.lines, b.shapes]);
  });
});
