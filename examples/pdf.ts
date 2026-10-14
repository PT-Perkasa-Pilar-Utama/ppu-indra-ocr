/**
 * Convert a PDF: the text layer where it can be trusted, OCR elsewhere.
 * Needs the optional peer dependency @embedpdf/pdfium.
 *
 *   bun examples/pdf.ts paper.pdf
 */

import { IndraOcrService } from "../src";
import { convertPdf } from "../src/pdf";

const file = process.argv[2] ?? `${import.meta.dir}/../tests/fixtures/pages/digital.pdf`;
const service = new IndraOcrService();
await service.initialize();

const pages = await convertPdf(service, await Bun.file(file).bytes(), { mode: "auto" });
for (const [i, page] of pages.entries()) {
  console.log(`--- page ${i + 1} (${page.source}: ${page.note})`);
  console.log(page.markdown);
}

await service.destroy();
