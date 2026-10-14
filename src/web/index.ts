// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/**
 * @module
 *
 * Indra OCR for browsers, web workers and browser extensions.
 *
 * @example
 * ```ts
 * import { IndraOcrService } from "ppu-indra-ocr/web";
 *
 * const service = new IndraOcrService();
 * await service.initialize();
 * const page = await service.convert(await file.arrayBuffer());
 * ```
 */

import type { ImageInput } from "../core/indra-ocr.service.js";
import { runOneShot } from "../core/one-shot.js";
import type { DetBox } from "../indra/det-post.js";
import type { Page } from "../indra/types.js";
import type { IndraOptions } from "../interface.js";
import { IndraOcrService } from "./indra-ocr.service.web.js";

export * from "../exports.js";
export { IndraOcrService };
export { getDefaultWebExecutionProviders, isWebGpuAvailable, isWebWorker } from "./platform.web.js";

/** OCR one image without managing a service: load the models, convert, release. */
export function ocr(image: ImageInput, options?: IndraOptions): Promise<Page> {
  return runOneShot(
    () => new IndraOcrService(options),
    (service) => service.convert(image)
  );
}

/** Detect the text lines of one image without managing a service. */
export function detect(image: ImageInput, options?: IndraOptions): Promise<DetBox[]> {
  return runOneShot(
    () => new IndraOcrService(options),
    (service) => service.detect(image)
  );
}
