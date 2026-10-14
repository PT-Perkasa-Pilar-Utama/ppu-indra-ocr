// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { rmSync } from "fs";
import { dirname } from "path";

import { cachePathFor } from "../src/processor/model-cache.js";
import { NodePlatformProvider } from "../src/processor/platform.node.js";

const receiptPath = `${import.meta.dir}/../assets/receipt.jpg`;
const receipt = await Bun.file(receiptPath).arrayBuffer();
const HTTP_NOT_FOUND = 404;

let server: ReturnType<typeof Bun.serve>;
let origin: string;

beforeAll(() => {
  server = Bun.serve({
    port: 0,
    fetch(req) {
      return new URL(req.url).pathname === "/receipt.jpg"
        ? new Response(receipt)
        : new Response("nope", { status: HTTP_NOT_FOUND });
    },
  });
  origin = `http://localhost:${server.port}`;
});

afterAll(() => {
  server.stop(true);
  // Downloads land in the real disk cache under a per-URL digest directory.
  for (const path of ["/receipt.jpg", "/missing.onnx"]) {
    rmSync(dirname(cachePathFor(`${origin}${path}`)), { recursive: true, force: true });
  }
});

describe("NodePlatformProvider.loadResource", () => {
  const platform = new NodePlatformProvider();

  test("returns an ArrayBuffer source untouched", async () => {
    expect(await platform.loadResource(receipt, `${origin}/unused`)).toBe(receipt);
  });

  test("reads a file path from disk", async () => {
    const loaded = await platform.loadResource(receiptPath, `${origin}/unused`);
    expect(loaded.byteLength).toBe(receipt.byteLength);
  });

  test("fetches an http URL and falls back to the default URL", async () => {
    const explicit = await platform.loadResource(`${origin}/receipt.jpg`, `${origin}/unused`);
    expect(explicit.byteLength).toBe(receipt.byteLength);

    const fallback = await platform.loadResource(undefined, `${origin}/receipt.jpg`);
    expect(fallback.byteLength).toBe(receipt.byteLength);
  });

  test("rejects a non-2xx response with the URL in the message", async () => {
    await expect(platform.loadResource(`${origin}/missing.onnx`, "")).rejects.toThrow(
      `${origin}/missing.onnx`
    );
  });
});

describe("NodePlatformProvider images", () => {
  const platform = new NodePlatformProvider();

  test("decodes image bytes into a canvas it recognizes, with RGBA pixels", async () => {
    const canvas = await platform.decodeImage(receipt);
    expect(platform.isCanvas(canvas)).toBe(true);
    expect(platform.isCanvas({ width: 4, height: 3 })).toBe(false);
    expect(platform.isCanvas(receipt)).toBe(false);
    const { data, width, height } = platform.pixels(canvas);
    expect(data.length).toBe(width * height * 4);
  });

  test("pixels refuses a non-canvas", () => {
    // SAFETY: deliberately the wrong type, to reach the runtime check.
    expect(() => platform.pixels({} as never)).toThrow("node canvas");
  });

  test("loads OpenCV.js on request", async () => {
    const cv = await platform.opencv();
    expect(typeof cv.resize).toBe("function");
    expect(platform.defaultEngine).toBe("opencv");
  });
});
