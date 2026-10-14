// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { readFile } from "fs/promises";
import * as ort from "onnxruntime-node";
import * as path from "path";
import { Canvas, ImageProcessor, cv } from "ppu-ocv";
import { CanvasProcessor } from "ppu-ocv/canvas";

import type { CoreCanvas, Cv, PlatformProvider, Rgba } from "../core/platform.js";
import { fetchAndCacheResource } from "./model-cache.js";

/** Node and Bun: onnxruntime-node, node canvas decoding, OpenCV.js, and a disk cache for downloads. */
export class NodePlatformProvider implements PlatformProvider<CoreCanvas> {
  public readonly ort: PlatformProvider["ort"] = ort;
  public readonly defaultEngine = "opencv" as const;

  public constructor(private readonly verbose = false) {}

  public isCanvas(image: unknown): image is CoreCanvas {
    return image instanceof Canvas;
  }

  public async decodeImage(bytes: ArrayBuffer): Promise<CoreCanvas> {
    return CanvasProcessor.prepareCanvas(bytes);
  }

  public pixels(canvas: CoreCanvas): Rgba {
    if (!(canvas instanceof Canvas)) throw new TypeError("expected a node canvas");
    return canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
  }

  public async loadResource(
    source: string | ArrayBuffer | undefined,
    defaultUrl: string
  ): Promise<ArrayBuffer> {
    if (source instanceof ArrayBuffer) return source;
    const target = source ?? defaultUrl;
    if (/^https?:\/\//.test(target)) return fetchAndCacheResource(target, this.verbose);
    const buf = await readFile(path.resolve(process.cwd(), target));
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }

  public readonly opencv = async (): Promise<Cv> => {
    await ImageProcessor.initRuntime();
    return cv;
  };
}
