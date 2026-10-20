// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/**
 * bun test preload: fill the model cache once, before any test file runs,
 * when models/ holds LFS pointers (CI). A cold-cache download inside a 5 s
 * test hook times out. With the real files in models/ this does nothing.
 */
import { testModelSources } from "./indra/golden.js";
import { IndraOcrService } from "../src/processor/indra-ocr.service.js";

import { MODEL_PRESETS } from "../src/model-catalogue.js";

for (const model of ["v1.1", "v3.1"] as const) {
  if (Object.keys(testModelSources(model)).length < MODEL_PRESETS[model].files.length)
    await IndraOcrService.downloadModels({ model, math: true });
}
