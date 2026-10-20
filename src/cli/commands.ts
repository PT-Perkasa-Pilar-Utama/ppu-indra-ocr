// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/**
 * Command implementations. Each owns one IndraOcrService lifecycle and throws
 * CliError on failure; the dispatcher maps that to an exit code. Results go
 * to stdout (or files); progress and logs go to stderr.
 */

import { mkdirSync, rmSync } from "node:fs";
import { basename, extname, join } from "node:path";

import { isPdf } from "../indra/document.js";
import type { Page } from "../indra/types.js";
import type * as PdfModule from "../pdf/index.js";
import type { ModelPreset } from "../model-catalogue.js";
import {
  DEFAULT_PRESET,
  MATH_FILES,
  MODEL_FILES,
  MODEL_PRESETS,
  defaultModelUrl,
} from "../model-catalogue.js";
import { CACHE_DIR } from "../processor/model-cache.js";
import { IndraOcrService } from "../processor/indra-ocr.service.js";
import {
  CliError,
  expandPatterns,
  isMissingLocalFile,
  loadImageInput,
  logStderr,
  writeOutput,
} from "./io.js";
import type { CliValues, OutputFormat } from "./options.js";
import { buildIndraOptions, buildPdfOptions, format, preset, str } from "./options.js";

const EXTENSIONS: Record<OutputFormat, string> = { md: ".md", text: ".txt", json: ".jsonl" };

function errMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

/** Pages in the chosen format: markdown, plain text, or one JSON object per page. */
export function renderPages(pages: readonly Page[], fmt: OutputFormat, pretty: boolean): string {
  if (fmt === "text") return pages.map((p) => p.text).join("\n");
  if (fmt === "json")
    return pages
      .map((p, page) => JSON.stringify({ page, ...p }, null, pretty ? 2 : undefined))
      .join("\n");
  return pages.map((p) => p.markdown ?? p.text).join("\n\n");
}

async function readPdf(
  service: IndraOcrService,
  bytes: ArrayBuffer,
  values: CliValues
): Promise<Page[]> {
  let pdf: typeof PdfModule;
  try {
    pdf = await import("../pdf/index.js");
  } catch (err) {
    throw new CliError(
      `PDF input needs the optional '@embedpdf/pdfium' package: npm install @embedpdf/pdfium (${errMessage(err)})`
    );
  }
  return pdf.convertPdf(service, bytes, buildPdfOptions(values));
}

async function withService<T>(
  values: CliValues,
  run: (service: IndraOcrService) => Promise<T>
): Promise<T> {
  const service = new IndraOcrService(buildIndraOptions(values));
  try {
    logStderr("Loading models...", Boolean(values.quiet));
    await service.initialize();
    return await run(service);
  } finally {
    await service.destroy();
  }
}

export async function runConvert(patterns: string[], values: CliValues): Promise<void> {
  if (!patterns.length) throw new CliError("convert takes one or more files", 2);
  const files = expandPatterns(patterns);
  const missing = files.filter(isMissingLocalFile);
  if (missing.length)
    throw new CliError(`No such file${missing.length > 1 ? "s" : ""}: ${missing.join(", ")}`);
  const fmt = format(values);
  const outDir = str(values, "out");
  if (outDir) mkdirSync(outDir, { recursive: true });
  const outputs: string[] = [];
  let failed = 0;
  await withService(values, async (service) => {
    for (const file of files) {
      try {
        const bytes = await loadImageInput(file);
        const pages = isPdf(bytes)
          ? await readPdf(service, bytes, values)
          : [await service.convert(bytes)];
        if (values.timings) {
          for (const [i, p] of pages.entries()) {
            process.stderr.write(
              `${file} p${i + 1}: ${p.source} ${Math.round(p.timings["total_ms"] ?? 0)} ms (${p.note})\n`
            );
          }
        }
        const rendered = renderPages(pages, fmt, Boolean(values.pretty));
        if (outDir) {
          writeOutput(rendered, join(outDir, basename(file, extname(file)) + EXTENSIONS[fmt]));
        } else {
          outputs.push(rendered);
        }
      } catch (err) {
        failed++;
        process.stderr.write(`${file}: ${errMessage(err)}\n`);
      }
    }
  });
  if (!outDir && outputs.length)
    writeOutput(outputs.join(fmt === "md" ? "\n\n" : "\n"), str(values, "output"));
  if (failed) throw new CliError(`${failed} of ${files.length} file(s) failed`);
}

export async function runDetect(images: string[], values: CliValues): Promise<void> {
  const [image] = images;
  if (!image || images.length !== 1) throw new CliError("detect takes exactly one image", 2);
  if (isMissingLocalFile(image)) throw new CliError(`No such file: ${image}`);
  const boxes = await withService(values, async (service) =>
    service.detect(await loadImageInput(image))
  );
  const out = boxes.map(([x0, y0, x1, y1, score]) => ({ box: [x0, y0, x1, y1], score }));
  writeOutput(JSON.stringify(out, null, values.pretty ? 2 : undefined), str(values, "output"));
}

export async function runDownloadModels(values: CliValues): Promise<void> {
  const model = preset(values) ?? DEFAULT_PRESET;
  // The script recognizers and language models load on first use: fetch
  // them all now, so a later run needs no network.
  await IndraOcrService.downloadModels({
    model,
    math: Boolean(values.math),
    lm: !values["no-lm"],
    verbose: Boolean(values.verbose),
  });
  logStderr(`Models (${model}) cached in ${CACHE_DIR}`, Boolean(values.quiet));
}

export function runClearCache(values: CliValues): void {
  rmSync(CACHE_DIR, { recursive: true, force: true });
  logStderr(`Removed ${CACHE_DIR}`, Boolean(values.quiet));
}

export function runModels(values: CliValues): void {
  // SAFETY: the keys of MODEL_PRESETS are its ModelPreset names.
  const names = Object.keys(MODEL_PRESETS) as ModelPreset[];
  const presets = names.map((name) => ({
    name,
    default: name === DEFAULT_PRESET,
    revision: MODEL_PRESETS[name].revision,
    tag: MODEL_PRESETS[name].tag,
    models: MODEL_PRESETS[name].files.map((role) => ({
      role,
      file: MODEL_FILES[role],
      url: defaultModelUrl(role, name),
      math: MATH_FILES.includes(role),
    })),
  }));
  if (format(values) === "json") {
    writeOutput(
      JSON.stringify(
        { default: DEFAULT_PRESET, cache: CACHE_DIR, presets },
        null,
        values.pretty ? 2 : undefined
      )
    );
    return;
  }
  const lines = presets.flatMap((p) => [
    `${p.name}${p.default ? " (default)" : ""}: revision ${p.revision} (tag ${p.tag})`,
    ...p.models.map((m) => `  ${m.file.padEnd(20)}${m.math ? " (--math)" : ""}`),
  ]);
  writeOutput(
    [
      "Indra OCR model presets (datalab-to/indra-ocr)",
      `Cache: ${CACHE_DIR}`,
      ...lines,
      "",
      "Weights: Datalab, modified OpenRAIL-M license with use restrictions.",
    ].join("\n"),
    str(values, "output")
  );
}
