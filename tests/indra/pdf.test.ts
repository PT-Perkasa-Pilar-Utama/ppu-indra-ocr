import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { init } from "@embedpdf/pdfium";
import type { WrappedPdfiumModule } from "@embedpdf/pdfium";
import { readFileSync } from "fs";
import { join } from "path";

import { assignRuns, pdfPage } from "../../src/indra/document.js";
import type { IndraPipeline } from "../../src/indra/pipeline.js";
import type { Box } from "../../src/indra/rec-preproc.js";
import type { TextLine } from "../../src/indra/textlayer.js";
import { orderBacktracks, pdfOrderCoherent, readTextLayer } from "../../src/indra/textlayer.js";
import { item } from "../../src/indra/util.js";
import { PdfiumDocument } from "../../src/pdf/pdfium.js";
import { event, field, loadCase, makePipeline, releaseSessions } from "./golden.js";

type GoldenRun = { bbox: number[]; text: string; chars: [string, number[]][] };

const ROOT = join(import.meta.dir, "..", "..");
const toBox = ([x0 = 0, y0 = 0, x1 = 0, y1 = 0]: readonly number[]): Box => [x0, y0, x1, y1];

let pdfium: WrappedPdfiumModule;
let pipeline: IndraPipeline;

beforeAll(async () => {
  const wasm = readFileSync(join(ROOT, "node_modules/@embedpdf/pdfium/dist/pdfium.wasm"));
  pdfium = await init({
    wasmBinary: wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength),
  });
  pdfium.PDFiumExt_Init();
  pipeline = await makePipeline();
});

function openCase(name: string): PdfiumDocument {
  return new PdfiumDocument(pdfium, readFileSync(join(ROOT, loadCase(name).input)));
}

describe("text layer matches the reference", () => {
  test("digital.pdf", () => {
    const c = loadCase("digital.auto");
    const want = event(c, "read_text_layer");
    const doc = openCase("digital.auto");
    const page = doc.page(0);
    try {
      const tl = readTextLayer(page, page.render(1), 1);
      expect(tl.trusted).toBe(field<boolean>(want, "trusted"));
      expect(tl.reason).toBe(field<string>(want, "reason"));
      expect(tl.stats).toEqual(field<typeof tl.stats>(want, "stats"));
      const runs = field<GoldenRun[]>(want, "runs");
      expect(tl.lines.map((l) => l.text)).toEqual(runs.map((r) => r.text));
      // pdfium builds differ (this wasm vs pypdfium2's 156.0.8076): glyph boxes move by up
      // to 0.57 pt measured (about 1.6 px at 200 dpi). Text, order and markdown still match.
      for (const [i, line] of tl.lines.entries()) {
        for (const [k, v] of line.bbox.entries())
          expect(Math.abs(v - item(item(runs, i).bbox, k))).toBeLessThan(0.75);
      }
    } finally {
      page.close();
      doc.close();
    }
  });
});

describe("pure PDF steps match the reference on its own inputs", () => {
  const c = loadCase("digital.auto");

  test("assignRuns", () => {
    const runs: TextLine[] = field<GoldenRun[]>(event(c, "read_text_layer"), "runs").map((r) => ({
      bbox: toBox(r.bbox),
      text: r.text,
      chars: r.chars.map(([ch, b]): [string, Box] => [ch, toBox(b)]),
    }));
    const ev = event(c, "assign_runs");
    const { toks, lost } = assignRuns(
      runs,
      field<number[][]>(ev, "boxes").map(toBox),
      field<number>(ev, "k")
    );
    const want = field<[string, number, number][][]>(ev, "toks");
    expect(lost).toBe(field<number>(ev, "lost"));
    expect(toks.map((tk) => tk.map((t) => [t[0], t[2]]))).toEqual(
      want.map((tk) => tk.map((t) => [t[0], t[2]]))
    );
    for (const [i, tk] of toks.entries()) {
      for (const [k, t] of tk.entries()) expect(t[1]).toBeCloseTo(item(item(want, i), k)[1], 2);
    }
  });

  test("pdfOrderCoherent", () => {
    const ev = event(c, "pdf_order_coherent");
    const boxes = field<number[][]>(ev, "boxes").map(toBox);
    expect(orderBacktracks(boxes)).toBe(field<number>(ev, "backtracks"));
    // The golden holds no logits; with backtracks under the limit the geometric check passes.
    expect(pdfOrderCoherent(boxes)).toBe(field<boolean>(ev, "value"));
  });
});

describe("PDF pages match the reference", () => {
  test.each([
    ["digital.auto", "auto"],
    ["digital.ocr", "ocr"],
    ["scanned.auto", "auto"],
  ] as const)("%s", async (name, mode) => {
    const want = item(loadCase(name).pages, 0);
    const doc = openCase(name);
    const page = doc.page(0);
    try {
      const got = await pdfPage(pipeline, page, { mode });
      expect<string>(got.source).toBe(want.source);
      expect(got.note).toBe(want.note);
      expect(got.lines.map((l) => l.text)).toEqual(want.lines.map((l) => l.text));
      expect(got.blocks.map((b) => [b.label, b.text])).toEqual(
        want.blocks.map((b) => [b.label, b.text])
      );
      expect(got.markdown).toBe(want.markdown);
    } finally {
      page.close();
      doc.close();
    }
  });
});

afterAll(releaseSessions);
