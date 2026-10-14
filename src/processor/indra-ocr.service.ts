// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { BaseIndraOcrService } from "../core/indra-ocr.service.js";
import type { CoreCanvas } from "../core/platform.js";
import type { IndraOptions } from "../interface.js";
import { BASE_FILES, MATH_FILES, defaultModelUrl } from "../model-catalogue.js";
import { fetchAndCacheResource } from "./model-cache.js";
import { NodePlatformProvider } from "./platform.node.js";

/**
 * Indra OCR for Node.js and Bun.
 *
 * @example
 * ```ts
 * import { IndraOcrService } from "ppu-indra-ocr";
 *
 * const service = new IndraOcrService();
 * await service.initialize();
 * const page = await service.convert(await Bun.file("scan.png").arrayBuffer());
 * console.log(page.markdown);
 * await service.destroy();
 * ```
 */
export class IndraOcrService extends BaseIndraOcrService<CoreCanvas> {
  public constructor(options: IndraOptions = {}) {
    super(new NodePlatformProvider(options.verbose), options);
  }

  /**
   * Download the default model files into the disk cache (~/.cache/ppu-indra-ocr)
   * without loading them. `math` adds the four math models and math_meta.json.
   */
  public static async downloadModels(
    options: { math?: boolean; verbose?: boolean } = {}
  ): Promise<void> {
    const roles = options.math ? [...BASE_FILES, ...MATH_FILES] : BASE_FILES;
    await Promise.all(
      roles.map((role) => fetchAndCacheResource(defaultModelUrl(role), options.verbose))
    );
  }
}
