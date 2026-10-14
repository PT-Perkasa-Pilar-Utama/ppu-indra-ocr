import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { createCanvas, loadImage } from "@napi-rs/canvas";
import { readFileSync } from "fs";
import { join } from "path";

import { IndraOcrService, detect, ocr } from "../src/index.js";
import { item } from "../src/indra/util.js";
import { loadCase, loadGray, testModelSources, tokenF1 } from "./indra/golden.js";

const ROOT = join(import.meta.dir, "..");
const bytes = (file: string): ArrayBuffer => {
  const b = readFileSync(join(ROOT, file));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

let service: IndraOcrService;

beforeAll(async () => {
  service = new IndraOcrService({ models: testModelSources() });
  await service.initialize();
});

afterAll(async () => {
  await service.destroy();
});

describe("IndraOcrService.convert from image files", () => {
  test("a lossless PNG matches the reference exactly", async () => {
    const page = await service.convert(bytes("tests/fixtures/pages/two-column.png"));
    expect(page.markdown).toBe(item(loadCase("two-column").pages, 0).markdown);
    expect(service.engine).toBe("opencv");
  });

  test("a JPEG stays within token-F1 0.98 of the reference", async () => {
    // Different JPEG decoders (libjpeg-turbo in cv2, the canvas decoder here)
    // give slightly different pixels, so this is not an exact match.
    const page = await service.convert(bytes("assets/receipt.jpg"));
    expect(tokenF1(page.text, item(loadCase("receipt").pages, 0).text)).toBeGreaterThanOrEqual(
      0.98
    );
  });

  test("accepts Uint8Array, a canvas, and gray pixels", async () => {
    const png = readFileSync(join(ROOT, "tests/fixtures/pages/table.png"));
    const want = item(loadCase("table").pages, 0).markdown;
    expect((await service.convert(new Uint8Array(png))).markdown).toBe(want);
    const img = await loadImage(png);
    const canvas = createCanvas(img.width, img.height);
    canvas.getContext("2d").drawImage(img, 0, 0);
    // SAFETY: @napi-rs/canvas is the node canvas ppu-ocv uses; the service checks it at runtime.
    expect((await service.convert(canvas as never)).markdown).toBe(want);
    expect((await service.convert(await loadGray("table.p0.gray.png"))).markdown).toBe(want);
  });

  test("rejects other inputs with a TypeError", async () => {
    // SAFETY: deliberately the wrong type, to reach the runtime check.
    await expect(service.convert("not an image" as never)).rejects.toThrow(TypeError);
  });
});

describe("IndraOcrService stages", () => {
  test("orient, straighten, detect, recognize and order chain like convert", async () => {
    const image = bytes("tests/fixtures/pages/rotated-90.png");
    const { turns } = await service.orient(image);
    expect(turns).toBe(1);
    const { image: upright, skew } = await service.straighten(image, turns);
    expect([upright.width, upright.height, skew]).toEqual([1700, 2200, 0]);
    const boxes = (await service.detect(upright)).map(
      ([x0, y0, x1, y1]) => [x0, y0, x1, y1] as const
    );
    const lines = await service.recognize(upright, boxes);
    expect(lines.length).toBe(boxes.length);
    const { order, cls } = await service.order(
      boxes,
      lines.map((l) => l.text),
      upright.width,
      upright.height
    );
    expect(order.toSorted((a, b) => a - b)).toEqual(boxes.map((_, i) => i));
    expect(cls).toBeDefined();
  });

  test("readMath needs math: true", async () => {
    await expect(service.readMath(await loadGray("math.p0.gray.png"), [])).rejects.toThrow(
      "math: true"
    );
  });
});

describe("IndraOcrService batches", () => {
  test("convertBatch keeps input order; settle reports a bad image", async () => {
    const good = bytes("assets/tilted.png");
    const pages = await service.convertBatch([good, good], { concurrency: 2 });
    expect(pages.map((p) => p.text)).toEqual([pages[0]?.text, pages[0]?.text]);
    const settled = await service.convertBatch([good, new ArrayBuffer(4)], { settle: true });
    expect(settled.map((r) => r.status)).toEqual(["fulfilled", "rejected"]);
    await expect(service.convertBatch([new ArrayBuffer(4)])).rejects.toBeInstanceOf(Error);
  });

  test("convertStream yields every result", async () => {
    const good = bytes("assets/tilted.png");
    const seen: number[] = [];
    let progress = 0;
    for await (const r of service.convertStream([good, good], {
      onProgress: (done) => (progress = done),
    })) {
      seen.push(r.index);
    }
    expect(seen.toSorted()).toEqual([0, 1]);
    expect(progress).toBe(2);
  });
});

describe("IndraOcrService lifecycle and options", () => {
  test("stages throw before initialize()", async () => {
    const fresh = new IndraOcrService();
    expect(fresh.isInitialized()).toBe(false);
    await expect(fresh.convert(new ArrayBuffer(8))).rejects.toThrow("initialize()");
  });

  test("a missing model file names the file", async () => {
    const bad = new IndraOcrService({
      models: { ...testModelSources(), det: "/no/such/det.onnx" },
    });
    await expect(bad.initialize()).rejects.toThrow(
      "could not load det.onnx from /no/such/det.onnx"
    );
  });

  test("math: true reads inline math; canvas-native runs without OpenCV", async () => {
    const math = new IndraOcrService({
      models: testModelSources(),
      math: true,
      engine: "canvas-native",
    });
    await math.initialize();
    try {
      expect(math.engine).toBe("canvas-native");
      const page = await math.convert(bytes("tests/fixtures/pages/math.png"));
      expect(page.markdown).toContain("$Q ( t ) = \\sum a _ { i } r ( t - i )$");
      const readings = await math.readMath(await loadGray("math.p0.gray.png"), []);
      expect(readings).toEqual([]);
    } finally {
      await math.destroy();
    }
    expect(math.isInitialized()).toBe(false);
  });

  test("one-shot ocr() and detect() load, run and release", async () => {
    const image = bytes("assets/tilted.png");
    const page = await ocr(image, { models: testModelSources() });
    // A color PNG: cv2 grays it through libpng's own rounding, so one token of
    // this deliberately poor sample can read differently.
    expect(tokenF1(page.text, item(loadCase("tilted").pages, 0).text)).toBeGreaterThanOrEqual(0.8);
    // detect() runs on the image as given (no deskew), so it finds more pieces than the 3 deskewed lines.
    expect((await detect(image, { models: testModelSources() })).length).toBe(
      (await service.detect(image)).length
    );
  });
});
