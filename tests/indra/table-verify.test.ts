import { beforeAll, describe, expect, test } from "bun:test";

import type { Cell } from "../../src/indra/layout-post.js";
import { pageMarkdown } from "../../src/indra/layout-post.js";
import { headsPlausible, tableHtmlHeads } from "../../src/indra/table-heads.js";
import type { Box } from "../../src/indra/rec-preproc.js";
import type { VerifierSet, VerifyFile, VerifyInput } from "../../src/indra/table-verify.js";
import { TableVerifier, blockFeatures } from "../../src/indra/table-verify.js";
import { loadCase, loadGray, modelBytes } from "./golden.js";
import type { Arr, PageJson, VerifyJson } from "./table-fixtures.js";
import { f32, maxDiff, readFixture } from "./table-fixtures.js";

/**
 * TableVerifier.apply and blockFeatures against the reference (indra-ocr 1.2.0),
 * on the three pages with tables and on crafted pages that reach the paths no
 * page does: a relabelled block, a decision that needs the ruling features, a
 * Table block without tsr_rel HTML, and the order model's table heads.
 */

type Crafted = {
  synthetic: VerifyJson[];
  tab: { boxes: Arr; texts: string[]; tab: Arr; plausible: boolean; html: string }[];
  tab_apply: VerifyJson[];
};

let weights: VerifyFile;
let verifiers: VerifierSet;

beforeAll(async () => {
  // SAFETY: table_verify.json ships with the models in the shape of VerifyFile.
  weights = JSON.parse(new TextDecoder().decode(await modelBytes("tableVerify"))) as VerifyFile;
  verifiers = TableVerifier.load(weights);
});

const boxesOf = (a: Arr): Box[] => {
  const v = f32(a);
  return Array.from({ length: v.length / 4 }, (_, i): Box => [
    v[i * 4] ?? 0,
    v[i * 4 + 1] ?? 0,
    v[i * 4 + 2] ?? 0,
    v[i * 4 + 3] ?? 0,
  ]);
};

async function inputOf(rec: VerifyJson): Promise<VerifyInput> {
  const cells = rec.cells
    ? new Map(
        Object.entries(rec.cells).map(([i, cs]): [number, Cell[]] => [
          Number(i),
          cs.map(([box, text]) => ({
            box: [box[0] ?? 0, box[1] ?? 0, box[2] ?? 0, box[3] ?? 0],
            text,
          })),
        ])
      )
    : undefined;
  return {
    boxes: boxesOf(rec.boxes),
    texts: rec.texts,
    order: rec.order,
    cls: f32(rec.cls),
    same: f32(rec.same),
    tab: rec.tab ? f32(rec.tab) : undefined,
    cells,
    tables: rec.tables ? new Map(Object.entries(rec.tables)) : undefined,
    gray: rec.gray ? await loadGray(rec.gray) : undefined,
  };
}

const verifierOf = (rec: VerifyJson & { threshold?: number }): TableVerifier => {
  if (rec.threshold !== undefined)
    return new TableVerifier({ ...weights, threshold: rec.threshold });
  const picked = rec.weights === "layer" ? verifiers.layer : verifiers.ocr;
  if (!picked) throw new Error("table_verify.json has no text-layer weights");
  return picked;
};

/** apply() gives the reference's relabelling, tables and class logits; blockFeatures its features. */
async function expectSame(rec: VerifyJson & { threshold?: number }): Promise<void> {
  const input = await inputOf(rec);
  const verifier = verifierOf(rec);
  const got = verifier.apply(input);
  expect(got.relabelled).toBe(rec.out.relabelled);
  expect(maxDiff(got.cls, f32(rec.out.cls))).toBe(0);
  expect(Object.fromEntries(got.tables ?? [])).toEqual(rec.out.tables ?? {});
  expect(got.cls === input.cls).toBe(!rec.out.relabelled);
  const { gray } = input;
  for (const b of rec.blocks) {
    const blk = b.key.split(",").map(Number);
    const features = blockFeatures({
      blk,
      boxes: input.boxes,
      texts: input.texts,
      cls: input.cls,
      html: b.html,
      gray,
      nTables: b.n_tables,
      headsOk: b.heads_ok,
      withRuling: false,
    });
    for (const [name, want] of Object.entries(b.features)) {
      if (name === "tsr" || name.startsWith("rule_")) continue;
      expect(features[name]).toBeCloseTo(want, 9);
    }
    const withTsr = { ...features, tsr: b.features["tsr"] ?? 0 };
    expect(verifier.decided(withTsr)).toBe(b.decided);
  }
}

describe.each(["table", "receipt", "sideways"])("%s", (name) => {
  const page = readFixture<PageJson>(`${name}.json`);
  test("the page's tables are kept, as the reference keeps them", async () => {
    for (const rec of page.verify) {
      expect(rec.out.relabelled).toBe(false);
      await expectSame(rec);
    }
  });
});

describe.each([
  ["table", "table"],
  ["receipt", "receipt"],
  ["sideways", "sideways.auto"],
])("%s markdown", (name, golden) => {
  const page = readFixture<PageJson>(`${name}.json`);
  test("a table block reads as the tsr_rel HTML, as in the reference's page", async () => {
    for (const rec of page.verify) {
      const input = await inputOf(rec);
      const md = (tables: VerifyInput["tables"]): string =>
        pageMarkdown(
          input.boxes,
          input.texts,
          input.order,
          f32(rec.out.cls),
          input.same,
          false,
          input.cells,
          tables
        );
      const want = loadCase(golden).pages[0]?.markdown;
      if (typeof want !== "string") throw new Error(`${golden}: the reference wrote no markdown`);
      expect(md(new Map(Object.entries(rec.out.tables ?? {})))).toBe(want);
    }
  });
});

describe("a verifier that calls every table false", () => {
  test("relabels the block and drops its tsr_rel table", async () => {
    const page = readFixture<PageJson & { verify_forced: (VerifyJson & { threshold: number })[] }>(
      "table.json"
    );
    expect(page.verify_forced.length).toBe(1);
    for (const rec of page.verify_forced) {
      expect(rec.tables && Object.keys(rec.tables).length).toBe(1);
      expect(rec.out.relabelled).toBe(true);
      await expectSame(rec);
    }
  });
});

describe("crafted pages", () => {
  const crafted = readFixture<Crafted>("verify-crafted.json");

  test("the pages reach every decision path", () => {
    const labels = crafted.synthetic.map((r) => r.label).toSorted();
    expect(labels).toEqual(["keep", "keep", "relabel", "relabel", "ruling_keep", "ruling_relabel"]);
  });

  test.each(crafted.synthetic.map((r, i) => [`${r.label} (${r.weights} weights)`, i] as const))(
    "%s",
    async (_, i) => {
      const rec = crafted.synthetic[i];
      if (!rec) throw new Error("no such case");
      await expectSame(rec);
    }
  );

  test("a decision that needs the ruling features gets them from the page", async () => {
    for (const rec of crafted.synthetic.filter((r) => r.label?.startsWith("ruling"))) {
      expect(rec.blocks[0]?.decided).toBeNull();
      expect(rec.blocks[0]?.rule).not.toBeNull();
    }
  });

  test("the order model's table heads: plausibility, HTML, and apply with `tab`", async () => {
    for (const t of crafted.tab) {
      const tab = f32(t.tab);
      expect(headsPlausible(boxesOf(t.boxes), tab)).toBe(t.plausible);
      expect(tableHtmlHeads(boxesOf(t.boxes), t.texts, tab)).toBe(t.html);
    }
    expect(crafted.tab.map((t) => t.plausible)).toEqual([true, false]);
    for (const rec of crafted.tab_apply) await expectSame(rec);
  });

  test("a tab of the wrong size is refused", async () => {
    const rec = crafted.tab_apply[0];
    if (!rec) throw new Error("no tab case");
    const input = await inputOf(rec);
    expect(() => verifiers.ocr.apply({ ...input, tab: new Float32Array(5) })).toThrow(RangeError);
  });
});
