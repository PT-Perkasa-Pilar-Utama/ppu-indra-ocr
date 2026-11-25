import { afterAll, beforeAll, describe, expect, test } from "bun:test";

import * as ort from "onnxruntime-node";
import { ImageProcessor, cv } from "ppu-ocv";

import { opencvImageOps } from "../../src/core/opencv-image-ops.js";
import type { Gray, ImageOps } from "../../src/indra/image.js";
import { TableReader, onnxMetadata, tsrRelInfo } from "../../src/indra/table-rel.js";
import type { TsrSessions } from "../../src/indra/table-rel.js";
import { loadGray, modelBytes, releaseSessions, session } from "./golden.js";
import type { PageJson, ReadJson } from "./table-fixtures.js";
import { linesOf, readFixture, replay } from "./table-fixtures.js";

/**
 * TableReader on the real tsr_rel, tsr_split and tsr_refine models against the
 * HTML of read_table in indra-ocr 1.2.0: the tables of three pages, and the paths
 * no page reaches (a mirrored table, other column spans, no split or refine model).
 */

let ops: ImageOps;
let sessions: TsrSessions;

beforeAll(async () => {
  await ImageProcessor.initRuntime();
  ops = opencvImageOps(cv);
  const rel = await session("tsrRel");
  sessions = {
    ...tsrRelInfo(rel, await modelBytes("tsrRel")),
    rel,
    split: await session("tsrSplit"),
    refine: await session("tsrRefine"),
  };
});
afterAll(releaseSessions);

/** A reader on the sessions the read used. */
function readerFor(args: ReadJson["args"], span: "geo" | "rel" | "rel2"): TableReader {
  const used = { ...sessions };
  if (!args.split) delete used.split;
  if (!args.refine) delete used.refine;
  return new TableReader(ort.Tensor, used, ops, span);
}

async function readAgain(
  gray: Gray,
  read: Pick<ReadJson, "args" | "decodes">
): Promise<string | null> {
  const { args, decodes } = read;
  return readerFor(args, args.col_spans).read(
    gray,
    linesOf(args),
    replay(args, decodes),
    args.mirror
  );
}

describe("the model", () => {
  test("its feature count and word-id buckets come from the session and the file", () => {
    expect(sessions.nFeat).toBe(41);
    expect(sessions.nVocab).toBe(4096);
  });

  test("onnxMetadata reads the metadata properties of the file bytes", async () => {
    const meta = onnxMetadata(await modelBytes("tsrRel"));
    expect(meta.get("n_vocab")).toBe("4096");
    expect(onnxMetadata(await modelBytes("tsrSplit")).size).toBe(0);
  });
});

describe.each(["table", "receipt", "sideways"])("%s", (name) => {
  const page = readFixture<PageJson>(`${name}.json`);
  test("each table block gets the reference's HTML", async () => {
    for (const { gray: file, reads, tables } of page.tables) {
      const gray = await loadGray(file);
      for (const read of reads) {
        const html = await readAgain(gray, read);
        expect(html).toBe(read.html);
        expect<string | undefined>(tables[read.key]).toBe(read.html ?? undefined);
      }
    }
  });
});

describe("table.png read other ways", () => {
  const page = readFixture<PageJson>("table.json");
  const gray = page.tables[0]?.gray ?? "";
  const variants = page.variants ?? [];

  test("the reference ran every variant", () => {
    expect(variants.map((v) => v.label)).toEqual([
      "rel",
      "geo",
      "rel2 mirror",
      "geo mirror",
      "rel2 without split",
      "rel2 without refine",
      "rel2 without both",
    ]);
  });

  test.each(variants.map((v) => [v.label, v] as const))("%s", async (_, variant) => {
    expect(await readAgain(await loadGray(gray), variant)).toBe(variant.html);
  });

  test("a table of fewer than two words is not read", async () => {
    const read = page.tables[0]?.reads[0];
    if (!read) throw new Error("no table in the fixture");
    const one = {
      ...read.args,
      lines: [{ ...(read.args.lines[0] ?? { box: [], toks: [], space: [] }), toks: [] }],
    };
    const html = await readerFor(one, "geo").read(
      await loadGray(gray),
      linesOf(one),
      replay(one, [])
    );
    expect(html).toBeNull();
  });
});
