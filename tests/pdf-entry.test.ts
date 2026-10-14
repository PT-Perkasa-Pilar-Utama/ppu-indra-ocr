import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

import { IndraOcrService } from "../src/index.js";
import type { IndraPipeline } from "../src/indra/pipeline.js";
import { item } from "../src/indra/util.js";
import { convertPdf, isPdf, loadPdfium } from "../src/pdf/index.js";
import { loadCase, testModelSources } from "./indra/golden.js";

const PAGES = join(import.meta.dir, "fixtures", "pages");
const pdf = (name: string): Uint8Array => readFileSync(join(PAGES, name));

let service: IndraOcrService;

beforeAll(async () => {
  service = new IndraOcrService({ models: testModelSources() });
  await service.initialize();
});

afterAll(async () => {
  await service.destroy();
});

describe("convertPdf", () => {
  test("reads a digital PDF's text layer", async () => {
    const pages = await convertPdf(service, pdf("digital.pdf"));
    expect(pages.length).toBe(1);
    const want = item(loadCase("digital.auto").pages, 0);
    expect(item(pages, 0).source).toBe("text");
    expect(item(pages, 0).markdown).toBe(want.markdown);
  });

  test("OCRs a scanned PDF, with a shared pdfium module", async () => {
    const pdfium = await loadPdfium();
    const pages = await convertPdf(service, new Uint8Array(pdf("scanned.pdf")).buffer, {
      pdfium,
      maxPages: 1,
    });
    expect(item(pages, 0).note).toBe(item(loadCase("scanned.auto").pages, 0).note);
  });

  test("opens an encrypted PDF only with its password", async () => {
    await expect(convertPdf(service, pdf("locked.pdf"))).rejects.toThrow("password-protected");
    const pages = await convertPdf(service, pdf("locked.pdf"), { password: "secret" });
    expect(item(pages, 0).text).toContain("locked with a password");
  });

  test("throws on bytes that are not a PDF, or a broken one", async () => {
    expect(isPdf(readFileSync(join(PAGES, "table.png")))).toBe(false);
    await expect(convertPdf(service, readFileSync(join(PAGES, "table.png")))).rejects.toThrow(
      "not a PDF"
    );
    await expect(convertPdf(service, new TextEncoder().encode("%PDF-1.7 broken"))).rejects.toThrow(
      "cannot open the PDF"
    );
  });

  test("a page that fails comes back as an error page", async () => {
    const failing = new Proxy(
      {},
      {
        get: (): never => {
          throw new Error("pipeline down");
        },
      }
    );
    // SAFETY: a stand-in whose every member throws, to reach the per-page catch.
    const pages = await convertPdf({ pipeline: failing as IndraPipeline }, pdf("digital.pdf"));
    expect(item(pages, 0).source).toBe("error");
    expect(item(pages, 0).note).toBe("Error: pipeline down");
  });
});
