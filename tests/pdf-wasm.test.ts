import { describe, expect, test } from "bun:test";

import { loadPdfium } from "../src/pdf/index.js";

describe("loadPdfium", () => {
  test("reads pdfium.wasm from PPU_INDRA_OCR_PDFIUM_WASM when it is set", async () => {
    const saved = process.env.PPU_INDRA_OCR_PDFIUM_WASM;
    process.env.PPU_INDRA_OCR_PDFIUM_WASM = "/nonexistent/pdfium.wasm";
    try {
      await expect(loadPdfium()).rejects.toThrow("/nonexistent/pdfium.wasm");
    } finally {
      if (saved === undefined) delete process.env.PPU_INDRA_OCR_PDFIUM_WASM;
      else process.env.PPU_INDRA_OCR_PDFIUM_WASM = saved;
    }
  });
});
