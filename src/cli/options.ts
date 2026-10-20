// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/** Maps parsed CLI flags onto IndraOptions and the PDF options. */

import { existsSync } from "node:fs";
import { join } from "node:path";
import type { ParseArgsConfig } from "node:util";

import type { PdfMode, PdfPageOptions } from "../indra/document.js";
import type { IndraOptions, ModelSources, ProcessingEngine } from "../interface.js";
import { MODEL_FILES, MODEL_PRESETS, isModelPreset } from "../model-catalogue.js";
import type { ModelFile, ModelPreset } from "../model-catalogue.js";
import { usageError } from "./io.js";

/** `parseArgs` option spec shared by every command. */
export const PARSE_OPTIONS: NonNullable<ParseArgsConfig["options"]> = {
  format: { type: "string" },
  out: { type: "string" },
  mode: { type: "string" },
  math: { type: "boolean" },
  "keep-furniture": { type: "boolean" },
  dpi: { type: "string" },
  "max-pages": { type: "string" },
  password: { type: "string" },
  "no-image-ocr": { type: "boolean" },
  engine: { type: "string" },
  model: { type: "string" },
  "no-lm": { type: "boolean" },
  "model-dir": { type: "string" },
  "execution-providers": { type: "string" },
  timings: { type: "boolean" },
  output: { type: "string", short: "o" },
  pretty: { type: "boolean" },
  quiet: { type: "boolean", short: "q" },
  verbose: { type: "boolean" },
  help: { type: "boolean", short: "h" },
  version: { type: "boolean", short: "v" },
};

/** Shape of `parseArgs().values` for the spec above. */
export type CliValues = Record<string, string | boolean | undefined>;

export type OutputFormat = "md" | "text" | "json";

/** A string flag's value, if given. */
export function str(values: CliValues, key: string): string | undefined {
  const raw = values[key];
  return typeof raw === "string" ? raw : undefined;
}

function num(values: CliValues, key: string): number | undefined {
  const raw = str(values, key);
  if (raw === undefined) return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0)
    usageError(`--${key} must be a non-negative number, got "${raw}"`);
  return n;
}

function oneOf<T extends string>(
  values: CliValues,
  key: string,
  allowed: readonly T[]
): T | undefined {
  const raw = str(values, key);
  if (raw === undefined) return undefined;
  const hit = allowed.find((a) => a === raw);
  if (!hit) usageError(`--${key} must be one of ${allowed.join(" | ")}`);
  return hit;
}

const ENGINES: readonly ProcessingEngine[] = ["opencv", "canvas-native"];
const MODES: readonly PdfMode[] = ["auto", "ocr", "text"];
const FORMATS: readonly OutputFormat[] = ["md", "text", "json"];

/** Output format; md by default. */
export function format(values: CliValues): OutputFormat {
  return oneOf(values, "format", FORMATS) ?? "md";
}

/**
 * The model files found in `--model-dir`, or undefined to use the preset's
 * repository. A file the folder lacks (a language model, a group recognizer)
 * is left out, so its stage is off rather than failing on first use.
 */
function modelDir(values: CliValues): ModelSources | undefined {
  const dir = str(values, "model-dir");
  if (dir === undefined) return undefined;
  const entries = Object.entries(MODEL_FILES).flatMap(([role, file]): [ModelFile, string][] =>
    existsSync(join(dir, file))
      ? // SAFETY: Object.entries of a Record<ModelFile, string> yields its ModelFile keys.
        [[role as ModelFile, join(dir, file)]]
      : []
  );
  return Object.fromEntries(entries);
}

/** The model preset flag. */
export function preset(values: CliValues): ModelPreset | undefined {
  const raw = str(values, "model");
  if (raw === undefined) return undefined;
  if (!isModelPreset(raw))
    usageError(`--model must be one of ${Object.keys(MODEL_PRESETS).join(" | ")}`);
  return raw;
}

/**
 * The engine flag. Slim standalone binaries ship without OpenCV
 * (PPU_BINARY_SLIM is baked in at build time), so they pin canvas-native.
 */
function engine(values: CliValues): ProcessingEngine | undefined {
  const chosen = oneOf(values, "engine", ENGINES);
  if (!process.env.PPU_BINARY_SLIM) return chosen;
  if (chosen === "opencv")
    usageError("this slim binary has no OpenCV: use --engine canvas-native or the full binary");
  return "canvas-native";
}

/** The service options from parsed flags. */
export function buildIndraOptions(values: CliValues): IndraOptions {
  const providers = str(values, "execution-providers");
  return {
    model: preset(values),
    models: modelDir(values),
    lm: values["no-lm"] ? false : undefined,
    math: Boolean(values.math),
    furniture: Boolean(values["keep-furniture"]),
    engine: engine(values),
    session: providers
      ? { executionProviders: providers.split(",").map((p) => p.trim()) }
      : undefined,
    verbose: Boolean(values.verbose),
  };
}

/** The PDF options from parsed flags. */
export function buildPdfOptions(
  values: CliValues
): PdfPageOptions & { maxPages?: number; password?: string } {
  return {
    mode: oneOf(values, "mode", MODES),
    dpi: num(values, "dpi"),
    imageOcr: !values["no-image-ocr"],
    maxPages: num(values, "max-pages"),
    password: str(values, "password"),
  };
}
