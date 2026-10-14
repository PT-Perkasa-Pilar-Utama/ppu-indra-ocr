// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/**
 * bun test preload: fill the model cache once, before any test file runs,
 * when models/ holds LFS pointers (CI). A cold-cache download inside a 5 s
 * test hook times out. With the real files in models/ this does nothing.
 */
import { testModelSources } from "./indra/golden.js";
import { IndraOcrService } from "../src/processor/indra-ocr.service.js";

if (Object.keys(testModelSources()).length < 12)
  await IndraOcrService.downloadModels({ math: true });
