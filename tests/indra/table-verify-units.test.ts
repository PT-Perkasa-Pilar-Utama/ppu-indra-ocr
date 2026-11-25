import { describe, expect, test } from "bun:test";

import { alignment, anyRun, parseTable, ruling } from "../../src/indra/table-block-features.js";
import type { Box } from "../../src/indra/rec-preproc.js";
import { loadGray } from "./golden.js";
import type { Arr } from "./table-fixtures.js";
import { f32, readFixture } from "./table-fixtures.js";

/**
 * The pure functions of table_verify.py on crafted inputs, against the
 * reference's outputs (scripts/golden/tables.py): table parsing, the
 * morphological open that finds ruling lines, ruling shares, line geometry.
 */

type Units = {
  parse_table: { html: string; rows: string[][] }[];
  any_run: { h: number; w: number; length: number; ink: number[]; out: boolean[] }[];
  alignment: { boxes: Arr; out: Record<string, number> }[];
  ruling: { gray: string; cases: { box: number[]; out: [number, number] }[] };
};

const units = readFixture<Units>("units.json");

describe("parseTable", () => {
  test.each(units.parse_table.map((c, i) => [i, c] as const))("case %d", (_, c) => {
    expect(parseTable(c.html)).toEqual(c.rows);
  });
});

describe("anyRun", () => {
  test.each(units.any_run.map((c, i) => [i, `${c.h}x${c.w} open ${c.length}`, c] as const))(
    "case %d (%s)",
    (_, __, c) => {
      expect(anyRun(Uint8Array.from(c.ink), c.h, c.w, c.length)).toEqual(c.out);
    }
  );

  test("the cases keep some rows and drop others", () => {
    const all = units.any_run.flatMap((c) => c.out);
    expect(all).toContain(true);
    expect(all).toContain(false);
  });
});

describe("alignment", () => {
  test.each(units.alignment.map((c, i) => [i, c] as const))("case %d", (_, c) => {
    const v = f32(c.boxes);
    const boxes = Array.from({ length: v.length / 4 }, (__, i): Box => [
      v[i * 4] ?? 0,
      v[i * 4 + 1] ?? 0,
      v[i * 4 + 2] ?? 0,
      v[i * 4 + 3] ?? 0,
    ]);
    const got = alignment(boxes);
    expect(Object.keys(got).toSorted()).toEqual(Object.keys(c.out).toSorted());
    for (const [k, want] of Object.entries(c.out))
      expect(Object.entries(got).find(([n]) => n === k)?.[1]).toBeCloseTo(want, 9);
  });
});

describe("ruling", () => {
  test("the share of the block crossed by long horizontal and vertical strokes", async () => {
    const gray = await loadGray(units.ruling.gray);
    for (const c of units.ruling.cases) {
      const [x0 = 0, y0 = 0, x1 = 0, y1 = 0] = c.box;
      expect(ruling(gray, [x0, y0, x1, y1])).toEqual(c.out);
    }
  });

  test("no page: no ruling", () => {
    expect(ruling(undefined, [0, 0, 10, 10])).toEqual([0, 0]);
  });
});
