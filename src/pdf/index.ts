// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/**
 * @module
 *
 * PDF input for Indra OCR: the PDF's own text layer where it can be trusted,
 * OCR elsewhere. Needs the optional peer dependency `@embedpdf/pdfium`.
 *
 * @example
 * ```ts
 * import { IndraOcrService } from "ppu-indra-ocr";
 * import { convertPdf } from "ppu-indra-ocr/pdf";
 *
 * const service = new IndraOcrService();
 * await service.initialize();
 * const pages = await convertPdf(service, pdfBytes);
 * console.log(pages.map((p) => p.markdown).join("\n\n"));
 * ```
 */

import { init } from "@embedpdf/pdfium";
import type { WrappedPdfiumModule } from "@embedpdf/pdfium";

import type { PdfPageOptions } from "../indra/document.js";
import { isPdf, pdfPage } from "../indra/document.js";
import type { IndraPipeline } from "../indra/pipeline.js";
import type { Page } from "../indra/types.js";
import { PdfiumDocument } from "./pdfium.js";

export type { PdfMode, PdfPageOptions } from "../indra/document.js";
export { isPdf } from "../indra/document.js";
export { PdfiumDocument, PdfiumPage } from "./pdfium.js";

/** The @embedpdf/pdfium release this package is tested with (its wasm is fetched from a CDN in browsers). */
export const PDFIUM_VERSION = "2.15.1";

export type ConvertPdfOptions = PdfPageOptions & {
  /** Read at most this many pages (0: all). */
  maxPages?: number;
  /** Password of an encrypted PDF. */
  password?: string;
  /** A pdfium module from loadPdfium(); loaded once and reused when omitted. */
  pdfium?: WrappedPdfiumModule;
};

let shared: Promise<WrappedPdfiumModule> | undefined;

/**
 * Load pdfium (WebAssembly). `wasm` is the pdfium.wasm bytes or URL; by
 * default Node reads it from the installed package and a browser fetches it
 * from jsDelivr.
 */
export async function loadPdfium(wasm?: ArrayBuffer | string): Promise<WrappedPdfiumModule> {
  const bytes = wasm instanceof ArrayBuffer ? wasm : await fetchWasm(wasm);
  const module = await init({ wasmBinary: bytes });
  module.PDFiumExt_Init();
  return module;
}

async function fetchWasm(url?: string): Promise<ArrayBuffer> {
  if (!url && globalThis.process?.versions?.node) {
    const { createRequire } = await import("module");
    const { readFile } = await import("fs/promises");
    // The standalone binaries embed pdfium.wasm and point here; node_modules is not there.
    const file = await readFile(
      process.env.PPU_INDRA_OCR_PDFIUM_WASM ||
        createRequire(import.meta.url).resolve("@embedpdf/pdfium/pdfium.wasm")
    );
    return file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
  }
  const src =
    url ?? `https://cdn.jsdelivr.net/npm/@embedpdf/pdfium@${PDFIUM_VERSION}/dist/pdfium.wasm`;
  const res = await fetch(src);
  if (!res.ok) throw new Error(`could not fetch pdfium.wasm from ${src}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

/** An empty page for one that failed, so one bad page does not end the document. */
function errorPage(err: unknown): Page {
  const name = err instanceof Error ? err.name : "Error";
  const message = err instanceof Error ? err.message : String(err);
  return {
    lines: [],
    text: "",
    markdown: "",
    blocks: [],
    source: "error",
    note: `${name}: ${message}`,
    rotation: 0,
    skew: 0,
    timings: { total_ms: 0 },
  };
}

/**
 * Every page of a PDF, in order. A page that fails comes back with
 * `source: "error"` and the reason in `note`; a file that cannot be opened
 * (not a PDF, corrupt, or locked) throws.
 */
export async function convertPdf(
  service: { readonly pipeline: IndraPipeline },
  bytes: ArrayBuffer | Uint8Array,
  options: ConvertPdfOptions = {}
): Promise<Page[]> {
  if (!isPdf(bytes)) throw new Error("not a PDF: no %PDF- header in the first 1024 bytes");
  const pdfium = options.pdfium ?? (await (shared ??= loadPdfium()));
  const doc = new PdfiumDocument(
    pdfium,
    bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes),
    options.password
  );
  try {
    const n = options.maxPages ? Math.min(doc.pageCount, options.maxPages) : doc.pageCount;
    const pages: Page[] = [];
    for (let i = 0; i < n; i++) {
      try {
        const page = doc.page(i);
        try {
          pages.push(await pdfPage(service.pipeline, page, options));
        } finally {
          page.close();
        }
      } catch (err) {
        pages.push(errorPage(err));
      }
    }
    return pages;
  } finally {
    doc.close();
  }
}
