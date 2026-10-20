import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { renderPages } from "../src/cli/commands.js";
import { expandPatterns, isMissingLocalFile, loadImageInput } from "../src/cli/io.js";
import { buildIndraOptions, buildPdfOptions, format } from "../src/cli/options.js";
import { main } from "../src/cli/run.js";
import { item } from "../src/indra/util.js";
import { MODEL_PRESETS } from "../src/model-catalogue.js";
import { loadCase, modelsDir, testModelSources } from "./indra/golden.js";

const PAGES = `${import.meta.dir}/fixtures/pages`;
const TABLE = `${PAGES}/table.png`;
const DIGITAL = `${PAGES}/digital.pdf`;

let workdir: string;
// Read models from models/v1.1 when it holds the real files, else from the cache.
const modelArgs = (): string[] =>
  Object.keys(testModelSources()).length === MODEL_PRESETS["v1.1"].files.length
    ? ["--model-dir", modelsDir()]
    : [];

/** Run `main` with stdout/stderr captured in-process. */
async function run(args: string[]): Promise<{ code: number; out: string; err: string }> {
  let out = "";
  let err = "";
  const origOut = process.stdout.write.bind(process.stdout);
  const origErr = process.stderr.write.bind(process.stderr);
  process.stdout.write = (chunk: string | Uint8Array): boolean => ((out += String(chunk)), true);
  process.stderr.write = (chunk: string | Uint8Array): boolean => ((err += String(chunk)), true);
  try {
    return { code: await main(args), out, err };
  } finally {
    process.stdout.write = origOut;
    process.stderr.write = origErr;
  }
}

beforeAll(() => {
  workdir = mkdtempSync(join(tmpdir(), "ppu-indra-cli-"));
});

afterAll(() => {
  if (workdir && existsSync(workdir)) rmSync(workdir, { recursive: true, force: true });
});

describe("option builders (no models)", () => {
  test("buildIndraOptions maps math, furniture, engine, providers and --model-dir", () => {
    const dir = join(workdir, "models");
    mkdirSync(dir, { recursive: true });
    for (const f of ["det.onnx", "math_meta.json"]) writeFileSync(join(dir, f), "");
    const opts = buildIndraOptions({
      math: true,
      "keep-furniture": true,
      engine: "canvas-native",
      "execution-providers": "cuda, cpu",
      "model-dir": dir,
    });
    expect(opts.math).toBe(true);
    expect(opts.furniture).toBe(true);
    expect(opts.engine).toBe("canvas-native");
    expect(opts.session?.executionProviders).toEqual(["cuda", "cpu"]);
    expect(opts.models?.det).toBe(join(dir, "det.onnx"));
    expect(opts.models?.mathMeta).toBe(join(dir, "math_meta.json"));
    // A file the folder lacks is left out, so its stage is off.
    expect(opts.models?.lm).toBeUndefined();
  });

  test("buildIndraOptions maps --model and --no-lm", () => {
    expect(buildIndraOptions({ model: "v3.1", "no-lm": true })).toMatchObject({
      model: "v3.1",
      lm: false,
    });
    expect(buildIndraOptions({}).model).toBeUndefined();
    expect(() => buildIndraOptions({ model: "v2" })).toThrow("--model must be one of");
  });

  test("buildPdfOptions maps the PDF flags", () => {
    expect(
      buildPdfOptions({
        mode: "ocr",
        dpi: "150",
        "max-pages": "2",
        password: "p",
        "no-image-ocr": true,
      })
    ).toEqual({
      mode: "ocr",
      dpi: 150,
      imageOcr: false,
      maxPages: 2,
      password: "p",
    });
  });

  test("bad flag values are usage errors", () => {
    expect(() => buildIndraOptions({ engine: "gpu" })).toThrow("--engine must be one of");
    expect(() => buildPdfOptions({ dpi: "-1" })).toThrow("--dpi must be a non-negative number");
    expect(() => format({ format: "html" })).toThrow("--format must be one of");
    expect(format({})).toBe("md");
  });

  test("renderPages writes md, text and one JSON object per page", () => {
    const page = {
      ...item(loadCase("table").pages, 0),
      source: "ocr" as const,
      vertical: false,
      lines: [],
      blocks: [],
      timings: {},
    };
    expect(renderPages([page, page], "md", false)).toBe(`${page.markdown}\n\n${page.markdown}`);
    expect(renderPages([page], "text", false)).toBe(page.text);
    const json = renderPages([page, page], "json", false)
      .split("\n")
      .map((l) => JSON.parse(l));
    expect(json.map((j) => j.page)).toEqual([0, 1]);
  });
});

describe("io helpers", () => {
  test("expandPatterns resolves globs and keeps plain paths", () => {
    expect(expandPatterns([`${PAGES}/two-*.png`])).toEqual([`${PAGES}/two-column.png`]);
    expect(expandPatterns(["missing.png", "https://x/y.png"])).toEqual([
      "missing.png",
      "https://x/y.png",
    ]);
    expect(() => expandPatterns([`${PAGES}/*.nothing`])).toThrow("No files match");
  });

  test("isMissingLocalFile and loadImageInput", async () => {
    expect(isMissingLocalFile("nope.png")).toBe(true);
    expect(isMissingLocalFile("https://x/y.png")).toBe(false);
    expect((await loadImageInput(TABLE)).byteLength).toBe(readFileSync(TABLE).byteLength);
    await expect(loadImageInput("nope.png")).rejects.toThrow("No such file");
  });
});

describe("commands", () => {
  test("help, version and usage errors", async () => {
    expect((await run(["help"])).out).toContain("ppu-indra-ocr - Indra OCR");
    expect((await run(["--version"])).out).toMatch(/^\d+\.\d+\.\d+/);
    expect((await run(["frobnicate"])).code).toBe(2);
    expect((await run(["--no-such-flag"])).code).toBe(2);
    expect((await run(["convert"])).code).toBe(2);
    expect((await run(["detect", TABLE, TABLE])).code).toBe(2);
  });

  test("convert writes markdown to stdout, matching the reference", async () => {
    const { code, out } = await run(["convert", TABLE, "-q", ...modelArgs()]);
    expect(code).toBe(0);
    expect(out.trim()).toBe(item(loadCase("table").pages, 0).markdown ?? "");
  });

  test("convert reads a PDF and writes one file per input with --out", async () => {
    const { code, err } = await run([
      "convert",
      DIGITAL,
      TABLE,
      "--out",
      workdir,
      "--format",
      "json",
      "--timings",
      "-q",
      ...modelArgs(),
    ]);
    expect(code).toBe(0);
    expect(err).toContain("digital.pdf p1: text");
    const pdf = JSON.parse(readFileSync(join(workdir, "digital.jsonl"), "utf8"));
    expect(pdf.markdown).toBe(item(loadCase("digital.auto").pages, 0).markdown);
    expect(existsSync(join(workdir, "table.jsonl"))).toBe(true);
  });

  test("a file that fails is reported and the exit code is 1", async () => {
    const bad = join(workdir, "not-an-image.png");
    writeFileSync(bad, "this is not a PNG");
    const { code, err } = await run(["convert", bad, "-q", ...modelArgs()]);
    expect(code).toBe(1);
    expect(err).toContain("not-an-image.png:");
    expect((await run(["convert", "missing.png", "-q"])).code).toBe(1);
  });

  test("detect prints line boxes as JSON", async () => {
    const { code, out } = await run(["detect", TABLE, "-q", ...modelArgs()]);
    expect(code).toBe(0);
    const boxes = JSON.parse(out) as { box: number[]; score: number }[];
    expect(boxes.length).toBeGreaterThan(10);
    expect(boxes[0]?.box.length).toBe(4);
  });

  test("models lists the presets; --format json names each revision", async () => {
    const out = (await run(["models"])).out;
    expect(out).toContain("v1.1 (default)");
    expect(out).toContain("rec_world.onnx");
    const json = JSON.parse((await run(["models", "--format", "json"])).out);
    expect(json.default).toBe("v1.1");
    const byName = new Map(json.presets.map((p: { name: string }) => [p.name, p]));
    expect(byName.get("v1.1")).toMatchObject({
      revision: "a1283c4b42a273c09480ad50e71461daff50fcc7",
      tag: "v1.1.0",
    });
    expect(byName.get("v3.1")).toMatchObject({
      revision: "d7c839c892bfe4b8993373db64c62f154d05d70b",
      tag: "v3.1",
    });
  });
});
