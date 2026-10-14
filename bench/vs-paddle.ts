// Indra OCR (this package) vs PaddleOCR (ppu-paddle-ocr): speed, memory, and
// accuracy against transcripts.
//
//   bun bench/vs-paddle.ts                                   # document pages
//   SROIE_DIR=path/to/ICDAR-2019-SROIE/data bun bench/vs-paddle.ts
//
// Without SROIE_DIR it reads the fixture pages (synthetic documents, with
// transcripts of the strings scripts/golden/make-fixtures.py draws) and the
// receipt in assets/. SROIE_DIR is the data/ folder of a
// zzzDavid/ICDAR-2019-SROIE clone (img/, box/): COUNT (default 50) receipts,
// every STEP-th (default 12). ROUNDS times each image (default 5 for pages, 1
// for SROIE; the median counts). Each library runs alone in a child process,
// one after the other, with its default options.
import { readdirSync } from "node:fs";
import { basename, join } from "node:path";

type Lib = "indra" | "paddle";
type Run = { name: string; ms: number; text: string };
type ChildResult = { initMs: number; peakMb: number; runs: Run[] };
/** One bench image and where its transcript comes from. */
type Item = { name: string; image: string; truth: () => Promise<string> };

const DIR = process.env.SROIE_DIR ?? "";
const COUNT = Number(process.env.COUNT ?? 50);
const STEP = Number(process.env.STEP ?? 12);
const ROUNDS = Number(process.env.ROUNDS ?? (DIR ? 1 : 5));
const ROOT = join(import.meta.dir, "..");

const PAGES: [image: string, truth: string][] = [
  ["tests/fixtures/pages/two-column.png", "bench/truth/two-column.txt"],
  ["tests/fixtures/pages/rotated-90.png", "bench/truth/two-column.txt"],
  ["tests/fixtures/pages/rotated-180.png", "bench/truth/two-column.txt"],
  ["tests/fixtures/pages/table.png", "bench/truth/table.txt"],
  ["tests/fixtures/pages/chinese.png", "bench/truth/chinese.txt"],
  ["assets/receipt.jpg", "assets/receipt-ground-truth.txt"],
];

/** Line transcripts of one SROIE receipt: the text after the 8 box coordinates. */
async function sroieTruth(csvPath: string): Promise<string> {
  const csv = await Bun.file(csvPath).text();
  return csv
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => line.split(",").slice(8).join(","))
    .join("\n");
}

function items(): Item[] {
  if (!DIR) {
    return PAGES.map(([image, truth]) => ({
      name: basename(image),
      image: join(ROOT, image),
      truth: () => Bun.file(join(ROOT, truth)).text(),
    }));
  }
  const names = readdirSync(join(DIR, "img"))
    .filter((f) => f.endsWith(".jpg"))
    .sort();
  return names
    .filter((_, i) => i % STEP === 0)
    .slice(0, COUNT)
    .map((name) => ({
      name,
      image: join(DIR, "img", name),
      truth: () => sroieTruth(join(DIR, "box", name.replace(/\.jpg$/, ".csv"))),
    }));
}

type Converter = { convert: (image: ArrayBuffer) => Promise<string>; close: () => Promise<void> };

async function load(lib: Lib): Promise<Converter> {
  if (lib === "indra") {
    const { IndraOcrService } = await import("../src/index.js");
    const service = new IndraOcrService();
    await service.initialize();
    return {
      convert: async (image) => (await service.convert(image)).text,
      close: () => service.destroy(),
    };
  }
  const { PaddleOcrService } = await import("ppu-paddle-ocr");
  const service = new PaddleOcrService();
  await service.initialize();
  return {
    convert: async (image) => (await service.recognize(image, { noCache: true })).text,
    close: () => service.destroy(),
  };
}

async function child(lib: Lib): Promise<ChildResult> {
  const list = items();
  const started = performance.now();
  const ocr = await load(lib);
  const initMs = performance.now() - started;
  const first = list[0];
  if (first) await ocr.convert(await Bun.file(first.image).arrayBuffer()); // warmup
  let peak = process.memoryUsage().rss;
  const runs: Run[] = [];
  for (const { name, image: path } of list) {
    const image = await Bun.file(path).arrayBuffer();
    const times: number[] = [];
    let text = "";
    for (let r = 0; r < ROUNDS; r++) {
      const t0 = performance.now();
      text = await ocr.convert(image);
      times.push(performance.now() - t0);
      peak = Math.max(peak, process.memoryUsage().rss);
    }
    runs.push({ name, ms: quantile(times, 0.5), text });
  }
  await ocr.close();
  return { initMs, peakMb: peak / 2 ** 20, runs };
}

/** F1 of two bags (multisets) of strings. */
function bagF1(got: string[], want: string[]): number {
  const count = (xs: string[]): Map<string, number> => {
    const m = new Map<string, number>();
    for (const x of xs) m.set(x, (m.get(x) ?? 0) + 1);
    return m;
  };
  const w = count(want);
  let hit = 0;
  for (const [t, n] of count(got)) hit += Math.min(n, w.get(t) ?? 0);
  const total = got.length + want.length;
  return total === 0 ? 1 : (2 * hit) / total;
}

/** Uppercase words. Order does not count; a missing space does. */
const words = (s: string): string[] => s.toUpperCase().split(/\s+/).filter(Boolean);
const tokenF1 = (got: string, want: string): number => bagF1(words(got), words(want));

/** Uppercase characters without whitespace: neither order nor spacing counts. */
const chars = (s: string): string[] => [...s.toUpperCase().replace(/\s+/g, "")];
const charF1 = (got: string, want: string): number => bagF1(chars(got), chars(want));

function levenshtein(a: string, b: string): number {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      const sub = (prev[j - 1] ?? 0) + (a[i - 1] === b[j - 1] ? 0 : 1);
      cur[j] = Math.min((prev[j] ?? 0) + 1, (cur[j - 1] ?? 0) + 1, sub);
    }
    prev = cur;
  }
  return prev[b.length] ?? 0;
}

/** CER with case and all whitespace removed. Still order-sensitive. */
function cer(got: string, want: string): number {
  const norm = (s: string): string => s.toUpperCase().replace(/\s+/g, "");
  const w = norm(want);
  return w.length ? levenshtein(norm(got), w) / w.length : 0;
}

function quantile(xs: number[], q: number): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))] ?? 0;
}

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);

async function spawnChild(lib: Lib): Promise<ChildResult> {
  const proc = Bun.spawn(["bun", import.meta.path, "--child", lib], {
    stdout: "pipe",
    stderr: "inherit",
  });
  const out = await new Response(proc.stdout).text();
  if ((await proc.exited) !== 0) throw new Error(`${lib} run failed`);
  const last = out.trim().split("\n").at(-1) ?? "{}";
  // SAFETY: the last stdout line is the JSON this file's child() wrote.
  return JSON.parse(last) as ChildResult;
}

async function main(): Promise<void> {
  const list = items();
  const truths = new Map(
    await Promise.all(list.map(async (x) => [x.name, await x.truth()] as const))
  );
  const set = DIR ? "SROIE receipts" : "document pages";
  console.log(`\n=== Indra OCR vs PaddleOCR on ${list.length} ${set}, ${ROUNDS} round(s) ===`);
  console.log(`${process.platform}/${process.arch}, bun ${Bun.version}, default options\n`);

  const results = new Map<Lib, ChildResult>();
  for (const lib of ["indra", "paddle"] as const) results.set(lib, await spawnChild(lib));

  const rows = [...results].map(([lib, r]) => {
    const ms = r.runs.map((x) => x.ms);
    const f1 = r.runs.map((x) => tokenF1(x.text, truths.get(x.name) ?? ""));
    const ch = r.runs.map((x) => charF1(x.text, truths.get(x.name) ?? ""));
    const c = r.runs.map((x) => cer(x.text, truths.get(x.name) ?? ""));
    return {
      lib,
      "init s": (r.initMs / 1000).toFixed(1),
      "median ms": quantile(ms, 0.5).toFixed(0),
      "p90 ms": quantile(ms, 0.9).toFixed(0),
      "peak RSS MB": r.peakMb.toFixed(0),
      "char-F1": mean(ch).toFixed(3),
      "token-F1": mean(f1).toFixed(3),
      CER: mean(c).toFixed(3),
    };
  });
  console.table(rows);

  const indra = results.get("indra")?.runs ?? [];
  const paddle = results.get("paddle")?.runs ?? [];
  if (indra.length <= 10) {
    console.table(
      indra.map((run, i) => {
        const want = truths.get(run.name) ?? "";
        const other = paddle[i] ?? { ms: 0, text: "" };
        return {
          page: run.name,
          "ms indra/paddle": `${run.ms.toFixed(0)} / ${other.ms.toFixed(0)}`,
          "char-F1": `${charF1(run.text, want).toFixed(3)} / ${charF1(other.text, want).toFixed(3)}`,
          "token-F1": `${tokenF1(run.text, want).toFixed(3)} / ${tokenF1(other.text, want).toFixed(3)}`,
          CER: `${cer(run.text, want).toFixed(3)} / ${cer(other.text, want).toFixed(3)}`,
        };
      })
    );
  }
  let indraWins = 0;
  let paddleWins = 0;
  for (const [i, run] of indra.entries()) {
    const want = truths.get(run.name) ?? "";
    const d = tokenF1(run.text, want) - tokenF1(paddle[i]?.text ?? "", want);
    if (d > 0.01) indraWins++;
    else if (d < -0.01) paddleWins++;
  }
  console.log(
    `token-F1 per image (margin 0.01): indra better ${indraWins}, paddle better ${paddleWins}, ` +
      `tie ${indra.length - indraWins - paddleWins}`
  );
  console.log(
    "char-F1: characters read, ignoring order and spacing. token-F1: words, so a\n" +
      "dropped space costs. CER: ignores case and whitespace but not order (SROIE\n" +
      "transcripts are not always in reading order). Peak RSS is sampled per page."
  );
}

const at = process.argv.indexOf("--child");
const lib = process.argv[at + 1];
if (at > 0 && (lib === "indra" || lib === "paddle")) console.log(JSON.stringify(await child(lib)));
else await main();
