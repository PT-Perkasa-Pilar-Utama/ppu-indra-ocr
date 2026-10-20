// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/**
 * Warm the on-disk model cache before the test suite runs (CI). Bun caps
 * hook timeouts at 5 s, and a cold-cache download inside a hook blows it.
 * Downloads every file of both model presets, math and language models
 * included: the goldens test v1.1 and v3.1.
 */

import { IndraOcrService } from "../src/processor/indra-ocr.service.js";

for (const model of ["v1.1", "v3.1"] as const) {
  await IndraOcrService.downloadModels({ model, math: true });
}
console.log("Model cache warmed (Indra OCR v1.1 and v3.1, math included).");
