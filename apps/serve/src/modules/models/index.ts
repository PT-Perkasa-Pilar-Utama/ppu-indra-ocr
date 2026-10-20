import type { Hono } from "hono";
import { MODEL_PRESETS } from "ppu-indra-ocr";
import { describeRoute } from "hono-openapi";
import { jsonResponse, success } from "../../core/api-response.js";
import { config } from "../../core/config.js";
import { modelsSchema } from "../../core/schemas.js";
import type { Env } from "../../core/types.js";

export function mount(app: Hono<Env>): void {
  app.get(
    "/v1/models",
    describeRoute({
      tags: ["System"],
      summary: "Available engines, the model preset and revision, and defaults",
      security: [{ Bearer: [] }],
      responses: { 200: jsonResponse("ok", modelsSchema) },
    }),
    (c) =>
      c.json(
        success(c, {
          engines: ["opencv", "canvas-native"],
          default: { engine: config.defaultEngine },
          model: config.model,
          revision: MODEL_PRESETS[config.model].revision,
          lm: config.lm,
          math: config.math,
          executionProviders: config.executionProviders,
        }),
        200
      )
  );
}
