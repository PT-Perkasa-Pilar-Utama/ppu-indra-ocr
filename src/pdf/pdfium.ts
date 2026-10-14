// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import type { WrappedPdfiumModule } from "@embedpdf/pdfium";

import type { Gray } from "../indra/image.js";
import type { PdfChar, PdfPageSource } from "../indra/textlayer.js";

const FPDFBITMAP_GRAY = 1;
const FPDF_ANNOT = 0x01;
const FPDF_GRAYSCALE = 0x08;
const FPDF_PAGEOBJ_TEXT = 1;
const FPDF_PAGEOBJ_FORM = 5;
const MAX_OBJECT_DEPTH = 4; // as the reference walks page objects

type Heaps = { HEAPU8: Uint8Array; HEAP32: Int32Array; HEAPF64: Float64Array };

/** The module's heap views. Read them after each call that may grow memory: growth replaces them. */
function heaps(p: WrappedPdfiumModule): Heaps {
  const m: unknown = p.pdfium;
  if (typeof m === "object" && m !== null && "HEAPU8" in m && "HEAP32" in m && "HEAPF64" in m) {
    // SAFETY: Emscripten puts these typed-array views on the module (checked above);
    // the package's typings leave them out.
    return m as Heaps;
  }
  throw new Error("the pdfium module exposes no heap views");
}

/** A PDF opened in pdfium (WebAssembly). Close it to free its memory. */
export class PdfiumDocument {
  private readonly doc: number;
  private readonly buffer: number;

  public constructor(
    private readonly p: WrappedPdfiumModule,
    bytes: Uint8Array,
    password = ""
  ) {
    this.buffer = p.pdfium.wasmExports.malloc(bytes.length);
    heaps(p).HEAPU8.set(bytes, this.buffer);
    this.doc = p.FPDF_LoadMemDocument(this.buffer, bytes.length, password);
    if (!this.doc) {
      p.pdfium.wasmExports.free(this.buffer);
      const err = p.FPDF_GetLastError();
      // FPDF_ERR_PASSWORD = 4
      throw new Error(
        err === 4
          ? "password-protected PDF (pass a password)"
          : `cannot open the PDF (pdfium error ${err})`
      );
    }
  }

  public get pageCount(): number {
    return this.p.FPDF_GetPageCount(this.doc);
  }

  /** Page `index` (0-based). Close it after use. */
  public page(index: number): PdfiumPage {
    const page = this.p.FPDF_LoadPage(this.doc, index);
    if (!page) throw new Error(`cannot load page ${index}`);
    return new PdfiumPage(this.p, page);
  }

  public close(): void {
    this.p.FPDF_CloseDocument(this.doc);
    this.p.pdfium.wasmExports.free(this.buffer);
  }
}

/** One loaded pdfium page, as the text-layer code reads it. */
export class PdfiumPage implements PdfPageSource {
  public constructor(
    private readonly p: WrappedPdfiumModule,
    private readonly handle: number
  ) {}

  public size(): readonly [number, number] {
    return [this.p.FPDF_GetPageWidthF(this.handle), this.p.FPDF_GetPageHeightF(this.handle)];
  }

  /** Grayscale render, as pypdfium2's render(scale, grayscale=True): white fill, annotations drawn. */
  public render(scale: number): Gray {
    const [w, h] = this.size();
    const width = Math.ceil(w * scale);
    const height = Math.ceil(h * scale);
    const bmp = this.p.FPDFBitmap_CreateEx(width, height, FPDFBITMAP_GRAY, 0, 0);
    if (!bmp) throw new Error(`cannot allocate a ${width}x${height} bitmap`);
    try {
      this.p.FPDFBitmap_FillRect(bmp, 0, 0, width, height, 0xffffffff);
      this.p.FPDF_RenderPageBitmap(
        bmp,
        this.handle,
        0,
        0,
        width,
        height,
        0,
        FPDF_GRAYSCALE | FPDF_ANNOT
      );
      const stride = this.p.FPDFBitmap_GetStride(bmp);
      const ptr = this.p.FPDFBitmap_GetBuffer(bmp);
      // Read the heap after rendering: a memory growth replaces the HEAPU8 view.
      const heap = heaps(this.p).HEAPU8;
      const data = new Uint8Array(width * height);
      for (let y = 0; y < height; y++)
        data.set(heap.subarray(ptr + y * stride, ptr + y * stride + width), y * width);
      return { data, width, height };
    } finally {
      this.p.FPDFBitmap_Destroy(bmp);
    }
  }

  public chars(): PdfChar[] {
    const tp = this.p.FPDFText_LoadPage(this.handle);
    if (!tp) return [];
    const out = this.p.pdfium.wasmExports.malloc(32); // left, right, bottom, top doubles
    try {
      const n = this.p.FPDFText_CountChars(tp);
      return Array.from({ length: Math.max(0, n) }, (_, i): PdfChar => {
        const ok = this.p.FPDFText_GetCharBox(tp, i, out, out + 8, out + 16, out + 24);
        const f64 = heaps(this.p).HEAPF64;
        const [l = 0, r = 0, b = 0, t = 0] = f64.subarray(out / 8, out / 8 + 4);
        return {
          unit: this.p.FPDFText_GetUnicode(tp, i),
          angle: this.p.FPDFText_GetCharAngle(tp, i),
          box: ok ? [l, b, r, t] : undefined,
        };
      });
    } finally {
      this.p.pdfium.wasmExports.free(out);
      this.p.FPDFText_ClosePage(tp);
    }
  }

  public pageToDevice(
    width: number,
    height: number,
    x: number,
    y: number
  ): readonly [number, number] {
    const out = this.p.pdfium.wasmExports.malloc(8);
    try {
      this.p.FPDF_PageToDevice(this.handle, 0, 0, width, height, 0, x, y, out, out + 4);
      const i32 = heaps(this.p).HEAP32;
      return [i32[out / 4] ?? 0, i32[out / 4 + 1] ?? 0];
    } finally {
      this.p.pdfium.wasmExports.free(out);
    }
  }

  public textRenderModes(): number[] {
    const modes: number[] = [];
    const walk = (count: number, get: (i: number) => number, level: number): void => {
      for (let i = 0; i < count; i++) {
        const obj = get(i);
        if (!obj) throw new Error("Failed to get pageobject.");
        const type = this.p.FPDFPageObj_GetType(obj);
        if (type === FPDF_PAGEOBJ_TEXT) modes.push(this.p.FPDFTextObj_GetTextRenderMode(obj));
        if (type === FPDF_PAGEOBJ_FORM && level < MAX_OBJECT_DEPTH - 1) {
          walk(
            this.p.FPDFFormObj_CountObjects(obj),
            (k) => this.p.FPDFFormObj_GetObject(obj, k),
            level + 1
          );
        }
      }
    };
    walk(
      this.p.FPDFPage_CountObjects(this.handle),
      (i) => this.p.FPDFPage_GetObject(this.handle, i),
      0
    );
    return modes;
  }

  public close(): void {
    this.p.FPDF_ClosePage(this.handle);
  }
}
