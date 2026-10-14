import * as v from "valibot";

const engine = v.optional(v.picklist(["opencv", "canvas-native"]));
const mode = v.optional(v.picklist(["auto", "ocr", "text"]));
const positiveInt = v.pipe(v.number(), v.integer(), v.minValue(1));

/** Per-request options shared by every input shape. `mode` and `maxPages` apply to PDFs. */
export const ocrOptionsSchema = v.object({
  engine,
  mode,
  maxPages: v.optional(positiveInt),
});
export type OcrOptions = v.InferOutput<typeof ocrOptionsSchema>;

// Multipart fields arrive as strings; coerce them to the JSON option types.
const formNumber = v.pipe(v.string(), v.transform(Number), v.number());

/** The same options read from multipart form fields. */
export const multipartOcrOptionsSchema = v.object({
  engine,
  mode,
  maxPages: v.optional(v.pipe(formNumber, v.integer(), v.minValue(1))),
});

// A 1x1 PNG as a data: URI - a valid, runnable example so Scalar's "Send"
// works out of the box (it decodes, finds no text, returns an empty result).
const EXAMPLE_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

const source = v.pipe(
  v.string(),
  v.minLength(1),
  v.description(
    "An image or PDF as a `data:` URI or an allowlisted https URL. Local paths are rejected."
  ),
  v.metadata({ example: EXAMPLE_IMAGE })
);

/** JSON body for single-image OCR. */
export const jsonOcrSchema = v.pipe(
  v.object({ ...ocrOptionsSchema.entries, source }),
  v.metadata({ ref: "OcrJsonRequest" })
);

/** JSON body for detection-only inference. */
export const jsonDetectSchema = v.pipe(
  v.object({ engine, source }),
  v.metadata({ ref: "DetectJsonRequest" })
);

/** JSON body for batch / async / stream OCR. */
export const batchOcrSchema = v.pipe(
  v.object({
    ...ocrOptionsSchema.entries,
    sources: v.pipe(
      v.array(v.pipe(v.string(), v.minLength(1))),
      v.minLength(1),
      v.description("data: URIs or https URLs.")
    ),
    concurrency: v.optional(positiveInt),
    settle: v.optional(v.boolean()),
  }),
  v.metadata({
    ref: "BatchRequest",
    example: { sources: [EXAMPLE_IMAGE], settle: true },
  })
);
export type BatchOcrBody = v.InferOutput<typeof batchOcrSchema>;

export const taskIdParamsSchema = v.object({
  id: v.pipe(v.string(), v.metadata({ example: crypto.randomUUID() })),
});

// --- Multipart bodies (documentation only - parsed manually) ---
// Plain OpenAPI objects: the file field has no schema-library shape, and these
// bodies never pass through a validator.

const stringEnum = (values: readonly string[]) => ({ type: "string" as const, enum: [...values] });
const multipartOptions = {
  engine: stringEnum(["opencv", "canvas-native"]),
  mode: stringEnum(["auto", "ocr", "text"]),
  maxPages: { type: "integer" as const, minimum: 1 },
};
const binaryFile = { type: "string" as const, format: "binary" };

/** Multipart body for single-image OCR. */
export const multipartOcrSchema = {
  type: "object" as const,
  properties: { file: binaryFile, ...multipartOptions },
  required: ["file"],
};

/** Multipart body for detection-only inference. */
export const multipartDetectSchema = {
  type: "object" as const,
  properties: { file: binaryFile, engine: multipartOptions.engine },
  required: ["file"],
};

// --- Response schemas ---

const box = v.tuple([v.number(), v.number(), v.number(), v.number()]);
const pageSchema = v.object({
  markdown: v.nullable(v.string()),
  text: v.string(),
  lines: v.array(v.object({ box, text: v.string(), conf: v.number() })),
  blocks: v.array(v.object({ label: v.string(), bbox: box, text: v.string() })),
  source: v.picklist(["ocr", "text", "hybrid", "error"]),
  note: v.string(),
  rotation: v.number(),
  skew: v.number(),
  timings: v.record(v.string(), v.number()),
});
export const ocrResultSchema = v.pipe(
  v.object({ pages: v.array(pageSchema) }),
  v.metadata({ ref: "OcrResult" })
);

export const detectResultSchema = v.pipe(
  v.object({ boxes: v.array(v.object({ box, score: v.number() })) }),
  v.metadata({ ref: "DetectResult" })
);

export const batchResultSchema = v.pipe(
  v.object({ results: v.array(v.unknown()) }),
  v.metadata({ ref: "BatchResult" })
);

export const taskAcceptedSchema = v.pipe(
  v.object({ taskId: v.string(), status: v.string() }),
  v.metadata({ ref: "TaskAccepted" })
);

export const taskStatusSchema = v.pipe(
  v.object({
    id: v.string(),
    status: v.picklist(["queued", "running", "done", "failed", "cancelled"]),
    updatedAt: v.number(),
  }),
  v.metadata({ ref: "TaskStatus" })
);

export const modelsSchema = v.pipe(
  v.object({
    engines: v.array(v.string()),
    default: v.object({ engine: v.string() }),
    revision: v.string(),
    math: v.boolean(),
    executionProviders: v.array(v.string()),
  }),
  v.metadata({ ref: "Models" })
);
