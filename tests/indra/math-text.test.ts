// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  MATH_MIN_CONF,
  SPLICE_MIN_LINE,
  SPLICE_MIN_SEG,
  acceptMath,
  gateFeatures,
  hasMath,
  hasStructure,
  latinLine,
  mathAccept,
  mathChars,
  mathGate,
  mathWrites,
  restoreAccents,
  sigDraft,
  sigReading,
  spliceProse,
  textLike,
  units,
} from "../../src/indra/math-text.js";
import type { MathAcceptHead, MathGateHead } from "../../src/indra/math-text.js";
import { GOLDEN_DIR } from "./golden.js";

type Heads = Record<string, { accept: MathAcceptHead | null; gate: MathGateHead | null }>;
type Variants = Record<string, (string | null)[]>;
type Crafted = {
  reading: string;
  draft: string;
  has_math: boolean;
  has_structure: boolean;
  accept_math: string | null;
  restore_accents: string;
  splice_prose: string;
  units: [string, number, number][];
  units_draft: [string, number, number][];
  latin_line: boolean;
  sig_reading: string;
  sig_draft: string;
  accept: Variants;
};
type Fixture = {
  constants: { MATH_MIN_CONF: number; SPLICE_MIN_SEG: number; SPLICE_MIN_LINE: number };
  heads: Heads;
  tokens: string;
  recorded_accept: {
    set: string;
    case: string;
    head: string;
    text: string;
    conf: number;
    draft: string;
    rconf: number;
    in_table: number;
    out: string | null;
  }[];
  recorded_gate: {
    set: string;
    case: string;
    head: string;
    probs: number[];
    drafts: string[];
    n_cand: number;
    source: string;
    keep: boolean[];
  }[];
  crafted: Crafted[];
  latin_lines: { text: string; latin: boolean }[];
  gate_features: { draft: string; args: [number, number, number, number, string]; x: number[] }[];
  gate_cases: {
    probs: number[];
    drafts: string[];
    n_cand: number;
    source: string;
    keep: Record<string, boolean[]>;
  }[];
  writes: { text: string; share: number; full: boolean; latin: boolean }[];
  text_like: { tokens: string[]; flags: boolean[]; in_math: boolean }[];
};

// Python 3.12 (Unicode 15.0.0) running indra-ocr 1.2.0: scripts/golden/math-text.py.
const fx: Fixture = JSON.parse(readFileSync(join(GOLDEN_DIR, "math-text.json"), "utf8"));

const acceptHead = (name: string): MathAcceptHead | null => {
  const h = fx.heads[name];
  if (!h) throw new Error(`fixture has no head "${name}"`);
  return h.accept;
};
const gateHead = (name: string): MathGateHead | null => {
  const h = fx.heads[name];
  if (!h) throw new Error(`fixture has no head "${name}"`);
  return h.gate;
};
const variant = (c: Crafted, name: string): (string | null)[] => {
  const v = c.accept[name];
  if (!v) throw new Error(`case has no variant "${name}"`);
  return v;
};

describe("math text rules on the recorded page events", () => {
  test("accept gives the recorded text on every math.accept event of the goldens", () => {
    expect(fx.recorded_accept.length).toBeGreaterThan(15);
    for (const e of fx.recorded_accept) {
      const got = mathAccept(acceptHead(e.head), e.text, e.conf, e.draft, e.rconf, e.in_table);
      expect([`${e.set}/${e.case}`, e.text, got]).toEqual([`${e.set}/${e.case}`, e.text, e.out]);
    }
    expect(fx.recorded_accept.some((e) => e.head === "v1.2" && e.out !== null)).toBe(true);
    expect(fx.recorded_accept.some((e) => e.head === "none" && e.out !== null)).toBe(true);
  });

  test("gate gives the recorded keep list on every math.gate event of the goldens", () => {
    expect(fx.recorded_gate.length).toBeGreaterThan(0);
    for (const e of fx.recorded_gate) {
      const got = mathGate(gateHead(e.head), e.probs, e.drafts, e.n_cand, e.source);
      expect([`${e.set}/${e.case}`, got]).toEqual([`${e.set}/${e.case}`, e.keep]);
    }
  });

  test("the constants are upstream's", () => {
    expect(MATH_MIN_CONF).toBe(fx.constants.MATH_MIN_CONF);
    expect(SPLICE_MIN_SEG).toBe(fx.constants.SPLICE_MIN_SEG);
    expect(SPLICE_MIN_LINE).toBe(fx.constants.SPLICE_MIN_LINE);
  });
});

describe("crafted readings and drafts", () => {
  test("has at least 150 cases, and the splice changes some of them", () => {
    expect(fx.crafted.length).toBeGreaterThanOrEqual(150);
    expect(fx.crafted.filter((c) => c.splice_prose !== c.restore_accents).length).toBeGreaterThan(
      50
    );
    expect(fx.crafted.filter((c) => c.accept_math !== null).length).toBeGreaterThan(50);
  });

  test("hasMath, hasStructure, acceptMath and restoreAccents", () => {
    for (const c of fx.crafted) {
      const at = [c.reading, c.draft];
      expect([at, hasMath(c.reading)]).toEqual([at, c.has_math]);
      expect([at, hasStructure(c.reading)]).toEqual([at, c.has_structure]);
      expect([at, acceptMath(c.reading, c.draft)]).toEqual([at, c.accept_math]);
      expect([at, restoreAccents(c.reading, c.draft)]).toEqual([at, c.restore_accents]);
    }
  });

  test("spliceProse", () => {
    for (const c of fx.crafted) {
      const at = [c.reading, c.draft];
      expect([at, spliceProse(c.reading, c.draft)]).toEqual([at, c.splice_prose]);
    }
  });

  test("units of a reading and of a draft", () => {
    for (const c of fx.crafted) {
      const at = [c.reading, c.draft];
      expect([at, units(c.reading)]).toEqual([at, c.units]);
      expect([at, units(c.draft)]).toEqual([at, c.units_draft]);
    }
  });

  test("sigReading, sigDraft and latinLine", () => {
    for (const c of fx.crafted) {
      const at = [c.reading, c.draft];
      expect([at, sigReading(c.reading)]).toEqual([at, c.sig_reading]);
      expect([at, sigDraft(c.draft)]).toEqual([at, c.sig_draft]);
      expect([at, latinLine(c.reading)]).toEqual([at, c.latin_line]);
    }
  });

  test("accept, with and without a learned head, a table flag and the prose splice", () => {
    const P1 = [0.95, 0.93, 0.0] as const;
    const P2 = [0.85, 0.55, 1.0] as const;
    for (const c of fx.crafted) {
      const at = [c.reading, c.draft];
      for (const name of ["none", "v1.2", "v1.2-tau0", "v1.2-no-table"]) {
        const head = acceptHead(name);
        const got = [P1, P2].map((p) => mathAccept(head, c.reading, p[0], c.draft, p[1], p[2]));
        expect([at, name, got]).toEqual([at, name, variant(c, name)]);
      }
      for (const [name, head] of [
        ["none-no-splice", null],
        ["v1.2-tau0-no-splice", acceptHead("v1.2-tau0")],
      ] as const) {
        const got = mathAccept(head, c.reading, P1[0], c.draft, P1[1], P1[2], false);
        expect([at, name, [got]]).toEqual([at, name, variant(c, name)]);
      }
    }
  });

  test("latinLine on lines of several scripts", () => {
    expect(fx.latin_lines.length).toBeGreaterThan(40);
    for (const l of fx.latin_lines) expect([l.text, latinLine(l.text)]).toEqual([l.text, l.latin]);
  });
});

describe("the read gate", () => {
  test("gateFeatures equals Python within 1e-12", () => {
    expect(fx.gate_features.length).toBeGreaterThan(100);
    for (const g of fx.gate_features) {
      const [prob, nFlagged, pmax, nCand, source] = g.args;
      const x = gateFeatures(prob, g.draft, nFlagged, pmax, nCand, source);
      expect(x.length).toBe(g.x.length);
      g.x.forEach((want, i) => {
        const at = [g.draft.slice(0, 20), g.args, i];
        expect([at, Math.abs((x[i] ?? Number.NaN) - want) <= 1e-12]).toEqual([at, true]);
      });
    }
  });

  test("mathGate keeps every line without a head and the learned lines with one", () => {
    for (const g of fx.gate_cases) {
      for (const name of ["none", "v1.2"]) {
        const got = mathGate(gateHead(name), g.probs, g.drafts, g.n_cand, g.source);
        expect([name, got]).toEqual([name, g.keep[name]]);
      }
    }
  });
});

describe("what MathRec writes", () => {
  test("mathWrites, with the full token set and with Latin tokens only", () => {
    const full = mathChars([...fx.tokens]);
    const latin = mathChars([...fx.tokens].filter((c) => c.charCodeAt(0) < 128));
    expect(fx.writes.length).toBeGreaterThan(100);
    expect(full.size).toBeGreaterThan(latin.size);
    for (const w of fx.writes) {
      const at = [w.text, w.share];
      expect([at, mathWrites(full, w.text, w.share)]).toEqual([at, w.full]);
      expect([at, mathWrites(latin, w.text, w.share)]).toEqual([at, w.latin]);
    }
  });

  test("textLike follows the $ spans token by token", () => {
    for (const run of fx.text_like) {
      let inMath = false;
      const flags: boolean[] = [];
      for (const t of run.tokens) {
        const step = textLike(t, inMath);
        inMath = step.inMath;
        flags.push(step.textLike);
      }
      expect([run.tokens, flags, inMath]).toEqual([run.tokens, run.flags, run.in_math]);
    }
  });
});
