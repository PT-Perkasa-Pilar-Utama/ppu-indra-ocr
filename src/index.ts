// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/**
 * @module
 *
 * Indra OCR for Node.js and Bun: page orientation, line detection,
 * recognition, reading order, layout blocks and markdown, on ONNX Runtime.
 *
 * @example
 * ```ts
 * import { IndraOcrService } from "ppu-indra-ocr";
 *
 * const service = new IndraOcrService();
 * await service.initialize();
 * const page = await service.convert(imageBytes);
 * console.log(page.markdown);
 * ```
 *
 * The model weights are Datalab's, under a modified OpenRAIL-M license with
 * use restrictions. See https://huggingface.co/datalab-to/indra-ocr.
 */

import { runOneShot } from "./core/one-shot.js";
import type { ImageInput } from "./core/indra-ocr.service.js";
import type { DetBox } from "./indra/det-post.js";
import type { Page } from "./indra/types.js";
import type { IndraOptions } from "./interface.js";
import { IndraOcrService } from "./processor/indra-ocr.service.js";

export * from "./exports.js";
export { IndraOcrService };

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
