// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import * as ort from "onnxruntime-web";
import { CanvasProcessor } from "ppu-ocv/canvas-web";
import { ImageProcessor, cv } from "ppu-ocv/web";

import type { CoreCanvas, Cv, PlatformProvider, Rgba } from "../core/platform.js";
import { fetchArrayBufferWithRetry } from "../utils.js";

/**
 * CDN copy of the ONNX Runtime WASM binaries, used when the host app picks no
 * location.
 *
 * The version comes from the runtime that is actually loaded rather than a pin,
 * so the binaries always match the JS glue that fetches them. A pin goes stale
 * on the next `onnxruntime-web` bump and then hands a 1.27 loader 1.26 binaries,
 * which fails at session-build time.
 */
function defaultWasmPaths(): string {
  const version = ort.env.versions.web ?? ort.env.versions.common;
  return `https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/`;
}

/**
 * True inside a Web Worker: dedicated, shared, or a Manifest V3 service worker.
 *
 * `WorkerGlobalScope` exists only in worker scopes, never on a page and never in
 * Node/Bun/Deno. `window` is the mirror image of it: absent in workers, which is
 * why probing for `window` alone reads a worker as a server runtime.
 */
export function isWebWorker(): boolean {
  // SAFETY: probing globalThis for a global that may not exist; the annotation
  // only says the property is optional, which is exactly what is being tested.
  return typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope === "function";
}

/**
 * Point ONNX Runtime at a CDN copy of its WASM binaries to avoid 404s on CDN or
 * unbundled usage. Applies to pages and workers alike; an explicit
 * `ort.env.wasm.wasmPaths` set by the host app always wins.
 *
 * Runs once on import. Exported so the environment probe stays testable.
 */
export function applyDefaultWasmPaths(): void {
  const inBrowser = typeof window !== "undefined" || isWebWorker();
  if (!inBrowser || ort.env.wasm.wasmPaths) return;
  ort.env.wasm.wasmPaths = defaultWasmPaths();
}

applyDefaultWasmPaths();

/** True when `navigator.gpu` is present and at least one adapter is available. */
export async function isWebGpuAvailable(): Promise<boolean> {
  if (typeof navigator === "undefined") return false;
  // SAFETY: `gpu` is absent from the DOM lib in some TS targets; the
  // intersection declares it optional and the next line tests for it.
  const nav = navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown | null> } };
  if (!nav.gpu || typeof nav.gpu.requestAdapter !== "function") return false;
  try {
    const adapter = await nav.gpu.requestAdapter();
    return adapter !== null && adapter !== undefined;
  } catch {
    return false;
  }
}

/** Returns `["webgpu", "wasm"]` when WebGPU is available, otherwise `["wasm"]`. */
export async function getDefaultWebExecutionProviders(): Promise<
  ort.InferenceSession.SessionOptions["executionProviders"]
> {
  if (await isWebGpuAvailable()) {
    return ["webgpu", "wasm"];
  }
  return ["wasm"];
}

/** Browsers and web workers: onnxruntime-web, canvas decoding, and OpenCV.js on request. */
export class WebPlatformProvider implements PlatformProvider<CoreCanvas> {
  // SAFETY: onnxruntime-web implements the onnxruntime-common Tensor and
  // InferenceSession contracts; its module type only adds web-specific members.
  public readonly ort = ort as unknown as PlatformProvider["ort"];
  // OpenCV.js is a multi-megabyte download: a page gets it only by asking for "opencv".
  public readonly defaultEngine = "canvas-native" as const;

  public isCanvas(image: unknown): image is CoreCanvas {
    return (
      typeof image === "object" &&
      image !== null &&
      "getContext" in image &&
      typeof image.getContext === "function"
    );
  }

  public async decodeImage(bytes: ArrayBuffer): Promise<CoreCanvas> {
    return CanvasProcessor.prepareCanvas(bytes);
  }

  public pixels(canvas: CoreCanvas): Rgba {
    const ctx: unknown = canvas.getContext("2d");
    if (!hasImageData(ctx)) throw new Error("canvas has no 2D context");
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }

  public async loadResource(
    source: string | ArrayBuffer | undefined,
    defaultUrl: string
  ): Promise<ArrayBuffer> {
    if (source instanceof ArrayBuffer) return source;
    // In a browser every string source is a URL. No referer, as in utils.ts.
    return fetchArrayBufferWithRetry(source ?? defaultUrl);
  }

  public readonly opencv = async (): Promise<Cv> => {
    // ppu-ocv/web does not fetch OpenCV.js; the page provides it on globalThis.cv.
    if (!("cv" in globalThis)) {
      throw new Error(
        'the "opencv" engine needs OpenCV.js on globalThis.cv: load @techstark/opencv-js ' +
          "with a <script> tag, or assign globalThis.cv, before initialize()"
      );
    }
    await ImageProcessor.initRuntime();
    return cv;
  };
}

type Ctx2D = { getImageData: (x: number, y: number, w: number, h: number) => Rgba };

function hasImageData(ctx: unknown): ctx is Ctx2D {
  return typeof ctx === "object" && ctx !== null && "getImageData" in ctx;
}
