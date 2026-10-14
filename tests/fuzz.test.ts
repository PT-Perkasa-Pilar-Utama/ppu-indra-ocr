// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/**
 * Dynamic analysis (fuzzing) of the untrusted-input boundary. `fast-check`
 * generates many random byte buffers at runtime and feeds them to the OCR
 * service's image-decode path; a crash, hang, or non-Error throw is a finding.
 * Assertions run throughout (the property uses `expect`).
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import fc from "fast-check";

import { IndraOcrService } from "../src/index.js";
import { testModelSources } from "./indra/golden.js";

describe("fuzz: OCR is robust to malformed image input", () => {
  let service: IndraOcrService;

  beforeAll(async () => {
    // canvas-native engine: no OpenCV.js init, faster per run.
    service = new IndraOcrService({ engine: "canvas-native", models: testModelSources() });
    await service.initialize();
  });

  afterAll(async () => {
    if (service) await service.destroy();
  });

  test("convert never crashes on arbitrary bytes", async () => {
    await fc.assert(
      fc.asyncProperty(fc.uint8Array({ maxLength: 8192 }), async (bytes) => {
        try {
          const result = await service.convert(bytes);
          // If random bytes happened to decode, the result must be well-formed.
          expect(Array.isArray(result.lines)).toBe(true);
        } catch (error) {
          // Malformed input must fail gracefully with a normal Error - never a
          // native crash, hang, or non-Error throw.
          expect(error).toBeInstanceOf(Error);
        }
      }),
      { numRuns: 100 }
    );
  }, 120000);

  test("convert rejects an empty buffer gracefully", async () => {
    await expect(service.convert(new ArrayBuffer(0))).rejects.toBeInstanceOf(Error);
  });
});
