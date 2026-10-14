// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import * as ort from "onnxruntime-react-native";
import { CanvasProcessor, getPlatform } from "ppu-ocv/canvas-mobile";

import type { CoreCanvas, PlatformProvider, Rgba } from "../core/platform.js";
import { fetchArrayBufferWithRetry } from "../utils.js";

/**
 * Default execution providers for React Native: CPU. NNAPI (Android) and
 * CoreML (iOS) can be opted into through `session.executionProviders`.
 */
export function getDefaultMobileExecutionProviders(): ort.InferenceSession.SessionOptions["executionProviders"] {
  return ["cpu"];
}

type Ctx2D = { getImageData: (x: number, y: number, w: number, h: number) => Rgba };

function hasImageData(ctx: unknown): ctx is Ctx2D {
  return typeof ctx === "object" && ctx !== null && "getImageData" in ctx;
}

/** React Native: onnxruntime-react-native and Skia canvases. No OpenCV: the engine is canvas-native. */
export class MobilePlatformProvider implements PlatformProvider<CoreCanvas> {
  // SAFETY: onnxruntime-react-native implements the onnxruntime-common Tensor
  // and InferenceSession contracts; its module type differs only in extras.
  public readonly ort = ort as unknown as PlatformProvider["ort"];
  public readonly defaultEngine = "canvas-native" as const;

  public isCanvas(image: unknown): image is CoreCanvas {
    return getPlatform().isCanvas(image);
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
    return fetchArrayBufferWithRetry(source ?? defaultUrl);
  }
}
