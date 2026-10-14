import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { getPlatform, setPlatform } from "ppu-ocv";

import { item } from "../src/indra/util.js";
import { loadCase, testModelSources, tokenF1 } from "./indra/golden.js";
import { installWebCanvas, uninstallWebCanvas } from "./web-canvas-polyfill.js";

/**
 * Unit tests for the web support module structure.
 * These tests verify that the web entry point exports are correctly wired up
 * and that the web services can be imported without pulling in Node-specific deps.
 */

// Importing the web entry (below, dynamically) calls ppu-ocv's process-global
// setPlatform(webPlatform). Save the platform before this file's tests and
// restore it after, so node test files that run later in the same `bun test`
// process aren't left on the web path (createImageBitmap is not defined).
let savedPlatform: ReturnType<typeof getPlatform>;
beforeAll(() => {
  savedPlatform = getPlatform();
});
afterAll(() => {
  setPlatform(savedPlatform);
});

describe("Web entry", () => {
  test("exports IndraOcrService, the one-shots and the shared helpers", async () => {
    const mod = await import("../src/web/index.js");
    expect(typeof mod.IndraOcrService).toBe("function");
    expect(typeof mod.ocr).toBe("function");
    expect(typeof mod.detect).toBe("function");
    expect(typeof mod.toMarkdown).toBe("function");
    expect(mod.MODEL_REVISION).toBe("d7c839c892bfe4b8993373db64c62f154d05d70b");
  });

  test("a new service is not initialized; destroy() is safe before initialize()", async () => {
    const { IndraOcrService } = await import("../src/web/index.js");
    const service = new IndraOcrService();
    expect(service.isInitialized()).toBe(false);
    await service.destroy();
    await expect(service.convert(new ArrayBuffer(8))).rejects.toThrow("initialize()");
  });
});

describe("Web code does not import Node-only modules", () => {
  const files = [
    "./src/web/index.ts",
    "./src/web/platform.web.ts",
    "./src/web/indra-ocr.service.web.ts",
    "./src/core/indra-ocr.service.ts",
    "./src/core/opencv-image-ops.ts",
    "./src/exports.ts",
  ];

  test.each(files)("%s", async (file) => {
    const content = await Bun.file(file).text();
    for (const mod of ["fs", "fs/promises", "path", "os", "onnxruntime-node", "@napi-rs/canvas"]) {
      expect(content).not.toContain(`from "${mod}"`);
    }
  });

  test("the pipeline core imports nothing platform-specific", async () => {
    for (const file of new Bun.Glob("src/indra/*.ts").scanSync(".")) {
      const content = await Bun.file(file).text();
      expect(content).not.toMatch(
        /from "(fs|path|os|onnxruntime-(node|web|react-native)|ppu-ocv[^"]*)"/
      );
    }
  });
});

describe("Web IndraOcrService reads a page", () => {
  test("canvas-native by default, within token-F1 0.98 of the reference", async () => {
    installWebCanvas();
    try {
      const { IndraOcrService } = await import("../src/web/index.js");
      // A browser cannot read file paths: hand the models over as bytes.
      const models = Object.fromEntries(
        Object.entries(testModelSources()).map(([role, file]) => [
          role,
          readFileSync(String(file)).buffer,
        ])
      );
      const service = new IndraOcrService({
        models: Object.keys(models).length ? models : undefined,
      });
      await service.initialize();
      try {
        expect(service.engine).toBe("canvas-native");
        const png = readFileSync("tests/fixtures/pages/table.png");
        const page = await service.convert(
          png.buffer.slice(png.byteOffset, png.byteOffset + png.byteLength)
        );
        expect(tokenF1(page.text, item(loadCase("table").pages, 0).text)).toBeGreaterThanOrEqual(
          0.98
        );
      } finally {
        await service.destroy();
      }
    } finally {
      uninstallWebCanvas();
    }
  }, 60000);
});

describe("WebGPU execution-provider detection", () => {
  test("isWebGpuAvailable is exported", async () => {
    const mod = await import("../src/web/index.js");
    expect(mod.isWebGpuAvailable).toBeDefined();
    expect(typeof mod.isWebGpuAvailable).toBe("function");
  });

  test("getDefaultWebExecutionProviders is exported", async () => {
    const mod = await import("../src/web/index.js");
    expect(mod.getDefaultWebExecutionProviders).toBeDefined();
    expect(typeof mod.getDefaultWebExecutionProviders).toBe("function");
  });

  test("isWebGpuAvailable returns false in Node/Bun (no navigator.gpu)", async () => {
    const { isWebGpuAvailable } = await import("../src/web/index.js");
    const result = await isWebGpuAvailable();
    expect(result).toBe(false);
  });

  test("getDefaultWebExecutionProviders falls back to ['wasm'] in Node/Bun", async () => {
    const { getDefaultWebExecutionProviders } = await import("../src/web/index.js");
    const providers = await getDefaultWebExecutionProviders();
    expect(providers).toEqual(["wasm"]);
  });

  test("isWebGpuAvailable detects WebGPU when navigator.gpu is present", async () => {
    const originalNavigator = (globalThis as { navigator?: unknown }).navigator;
    const mockAdapter = { features: new Set() };
    (globalThis as { navigator?: unknown }).navigator = {
      gpu: {
        requestAdapter: async () => mockAdapter,
      },
    };

    try {
      const { isWebGpuAvailable } = await import("../src/web/index.js");
      const result = await isWebGpuAvailable();
      expect(result).toBe(true);
    } finally {
      if (originalNavigator === undefined) {
        delete (globalThis as { navigator?: unknown }).navigator;
      } else {
        (globalThis as { navigator?: unknown }).navigator = originalNavigator;
      }
    }
  });

  test("isWebGpuAvailable returns false when requestAdapter rejects", async () => {
    const originalNavigator = (globalThis as { navigator?: unknown }).navigator;
    (globalThis as { navigator?: unknown }).navigator = {
      gpu: {
        requestAdapter: async () => {
          throw new Error("nope");
        },
      },
    };

    try {
      const { isWebGpuAvailable } = await import("../src/web/index.js");
      const result = await isWebGpuAvailable();
      expect(result).toBe(false);
    } finally {
      if (originalNavigator === undefined) {
        delete (globalThis as { navigator?: unknown }).navigator;
      } else {
        (globalThis as { navigator?: unknown }).navigator = originalNavigator;
      }
    }
  });

  test("isWebGpuAvailable returns false when requestAdapter returns null (e.g., no hardware)", async () => {
    const originalNavigator = (globalThis as { navigator?: unknown }).navigator;
    (globalThis as { navigator?: unknown }).navigator = {
      gpu: {
        requestAdapter: async () => null,
      },
    };

    try {
      const { isWebGpuAvailable } = await import("../src/web/index.js");
      const result = await isWebGpuAvailable();
      expect(result).toBe(false);
    } finally {
      if (originalNavigator === undefined) {
        delete (globalThis as { navigator?: unknown }).navigator;
      } else {
        (globalThis as { navigator?: unknown }).navigator = originalNavigator;
      }
    }
  });

  test("getDefaultWebExecutionProviders prefers WebGPU first when available", async () => {
    const originalNavigator = (globalThis as { navigator?: unknown }).navigator;
    (globalThis as { navigator?: unknown }).navigator = {
      gpu: {
        requestAdapter: async () => ({ features: new Set() }),
      },
    };

    try {
      const { getDefaultWebExecutionProviders } = await import("../src/web/index.js");
      const providers = await getDefaultWebExecutionProviders();
      expect(providers).toEqual(["webgpu", "wasm"]);
    } finally {
      if (originalNavigator === undefined) {
        delete (globalThis as { navigator?: unknown }).navigator;
      } else {
        (globalThis as { navigator?: unknown }).navigator = originalNavigator;
      }
    }
  });
});
