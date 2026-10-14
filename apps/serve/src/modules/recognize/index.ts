import type { Hono } from "hono";
import { describeRoute, resolver } from "hono-openapi";
import { errorResponse, jsonResponse, success } from "../../core/api-response.js";
import { readSingle } from "../../core/input.js";
import { runOcr } from "../../core/runner.js";
import { jsonOcrSchema, ocrResultSchema } from "../../core/schemas.js";
import type { Env } from "../../core/types.js";

// This endpoint accepts BOTH multipart and JSON. A body validator is bound to
// one content type, so the route is documented here and parses its body
// manually in readSingle().
export function mount(app: Hono<Env>): void {
  app.post(
    "/v1/ocr",
    describeRoute({
      tags: ["OCR"],
      summary: "Read an image or a PDF",
      description:
        "multipart/form-data with a `file` field, or JSON `{ source, ...options }`. An image gives one page; a PDF gives one page each (the text layer where it can be trusted, OCR elsewhere).",
      security: [{ Bearer: [] }],
      requestBody: {
        required: true,
        content: {
          "multipart/form-data": { schema: { $ref: "#/components/schemas/OcrMultipartRequest" } },
          "application/json": { schema: resolver(jsonOcrSchema) },
        },
      },
      responses: {
        200: jsonResponse(
          "Pages with markdown, text, lines and layout blocks. `metadata` carries `speed` (seconds).",
          ocrResultSchema
        ),
        400: errorResponse("Invalid input, unsupported image type, or source error."),
        413: errorResponse("Image exceeds MAX_UPLOAD_BYTES."),
        429: errorResponse("Rate limit or inference queue full."),
      },
    }),
    async (c) => {
      const { image, opts } = await readSingle(c);
      const { result, meta } = await runOcr(image, opts);
      return c.json(
        success(c, result, {
          speed: meta.ms / 1000,
          pages: result.pages.length,
          engine: meta.engine,
        })
      );
    }
  );
}
