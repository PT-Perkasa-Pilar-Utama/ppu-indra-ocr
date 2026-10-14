// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/**
 * Warm the on-disk model cache before the test suite runs (CI). Bun caps
 * hook timeouts at 5 s, and a cold-cache download inside a hook blows it.
 * Downloads every Indra model file, math included.
 */

import { IndraOcrService } from "../src/processor/indra-ocr.service.js";

await IndraOcrService.downloadModels({ math: true });
console.log("Model cache warmed (Indra OCR, math included).");
