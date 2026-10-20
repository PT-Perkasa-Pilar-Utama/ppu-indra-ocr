// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { BaseIndraOcrService } from "../core/indra-ocr.service.js";
import type { CoreCanvas } from "../core/platform.js";
import type { IndraOptions } from "../interface.js";
import type { ModelPreset } from "../model-catalogue.js";
import { DEFAULT_PRESET, defaultModelUrl, presetFiles } from "../model-catalogue.js";
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
   * Download a preset's model files (default v1.1) into the disk cache
   * (~/.cache/ppu-indra-ocr) without loading them, so later runs need no
   * network. `math` adds the four math models and math_meta.json; `lm: false`
   * leaves out the language models.
   */
  public static async downloadModels(
    options: { model?: ModelPreset; math?: boolean; lm?: boolean; verbose?: boolean } = {}
  ): Promise<void> {
    const preset = options.model ?? DEFAULT_PRESET;
    for (const role of presetFiles(preset, options)) {
      await fetchAndCacheResource(defaultModelUrl(role, preset), options.verbose);
    }
  }
}
