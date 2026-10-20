import { describe, expect, test } from "bun:test";

import {
  DEFAULT_PRESET,
  MATH_FILES,
  MODEL_BASE_URL,
  MODEL_PRESETS,
  MODEL_REVISION,
  defaultModelUrl,
  isModelPreset,
  presetFiles,
} from "../src/model-catalogue.js";

describe("model presets", () => {
  test("v1.1 is the default and names MODEL_REVISION and MODEL_BASE_URL", () => {
    expect(DEFAULT_PRESET).toBe("v1.1");
    expect(MODEL_REVISION).toBe(MODEL_PRESETS["v1.1"].revision);
    expect(MODEL_BASE_URL).toBe(MODEL_PRESETS["v1.1"].baseUrl);
    expect(MODEL_BASE_URL).toContain(MODEL_REVISION);
  });

  test("isModelPreset knows the presets only", () => {
    expect(isModelPreset("v1.1")).toBe(true);
    expect(isModelPreset("v3.1")).toBe(true);
    expect(isModelPreset("v2")).toBe(false);
    expect(isModelPreset("toString")).toBe(false);
  });

  test("presetFiles leaves out math unless asked, and the language models when lm is off", () => {
    const base = presetFiles("v1.1");
    expect(base).toContain("recRtl");
    expect(base).toContain("lm");
    expect(base.some((r) => MATH_FILES.includes(r))).toBe(false);
    expect(presetFiles("v1.1", { math: true })).toEqual(expect.arrayContaining([...MATH_FILES]));
    expect(presetFiles("v1.1", { lm: false })).not.toContain("lmCjk");
    // v3.1 has no right-to-left recognizer and no language models.
    expect(presetFiles("v3.1")).toEqual([
      "det",
      "rec",
      "script",
      "orient",
      "order",
      "meta",
      "recCjk",
    ]);
  });

  test("defaultModelUrl uses the preset's pinned revision and refuses a file it lacks", () => {
    expect(defaultModelUrl("rec", "v3.1")).toBe(`${MODEL_PRESETS["v3.1"].baseUrl}/rec.onnx`);
    expect(defaultModelUrl("lm")).toBe(`${MODEL_BASE_URL}/lm.npz`);
    expect(() => defaultModelUrl("recRtl", "v3.1")).toThrow(
      "model preset v3.1 has no rec_rtl.onnx"
    );
  });
});
