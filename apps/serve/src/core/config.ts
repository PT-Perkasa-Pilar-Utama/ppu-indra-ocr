import * as v from "valibot";

const toArray = (val: string | undefined): string[] =>
  (val ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

const toBoolean = (val: string | undefined): boolean =>
  val ? ["true", "1", "yes", "on"].includes(val.toLowerCase()) : false;

/** Env values arrive as strings; parse them as numbers and reject NaN. */
const num = v.pipe(v.union([v.string(), v.number()]), v.transform(Number), v.number());
const positiveInt = v.pipe(num, v.integer(), v.minValue(1));
const nonNegativeInt = v.pipe(num, v.integer(), v.minValue(0));
const str = (fallback: string) => v.optional(v.string(), fallback);
const int = (fallback: number) => v.optional(positiveInt, fallback);

/** Raw environment schema. Every knob is documented in `.env.example`. */
const EnvSchema = v.object({
  API_ENV: v.optional(v.picklist(["development", "production"]), "development"),
  PORT: int(8080),
  HOST: str("0.0.0.0"),

  // Security
  SECRET_KEY: v.optional(v.string()),
  IP_WHITE_LIST: str("*"),
  IP_DENY_LIST: str(""),
  CORS_ORIGINS: str("*"),
  DOCS_ENABLED: str("true"),

  // Rate limiting (fixed window, per client IP)
  RATE_LIMIT_ENABLED: str("true"),
  RATE_LIMIT_PER_WINDOW: int(120),
  RATE_LIMIT_WINDOW_SECONDS: int(60),

  // Request limits
  REQUEST_TIMEOUT_SECONDS: int(30),
  MAX_UPLOAD_BYTES: int(10 * 1024 * 1024),
  MAX_IMAGE_PIXELS: int(40_000_000),
  MAX_BATCH_IMAGES: int(32),

  // OCR engine / models
  EXECUTION_PROVIDERS: str("cpu"),
  DEFAULT_ENGINE: v.optional(v.picklist(["opencv", "canvas-native"]), "opencv"),
  MODEL_DIR: v.optional(v.string()),
  MATH: str("false"),
  KEEP_FURNITURE: str("false"),
  PDF_DPI: int(200),
  MAX_PDF_PAGES: int(50),

  // Inference backpressure / async tasks
  MAX_CONCURRENCY: v.optional(nonNegativeInt, 0),
  MAX_QUEUE_DEPTH: int(100),
  TASK_TTL_SECONDS: int(600),

  // Source fetching (SSRF allowlist of https hosts)
  SOURCE_URL_ALLOWLIST: str(""),
});

function build(raw: NodeJS.ProcessEnv) {
  const parsed = v.safeParse(EnvSchema, raw);
  if (!parsed.success) {
    console.error(
      "Invalid environment variables:",
      JSON.stringify(v.flatten(parsed.issues).nested, null, 2)
    );
    process.exit(1);
  }
  const e = parsed.output;

  const executionProviders = toArray(e.EXECUTION_PROVIDERS);
  const usesAccelerator = executionProviders.some((p) => p !== "cpu" && p !== "wasm");
  // 0 = auto: one page at a time. Each page allocates large tensors, and an
  // accelerator serializes device work anyway. Raise it on big hosts.
  const autoConcurrency = 1;

  return {
    apiEnv: e.API_ENV,
    port: e.PORT,
    host: e.HOST,

    secretKey: e.SECRET_KEY,
    ipWhiteList: toArray(e.IP_WHITE_LIST),
    ipDenyList: toArray(e.IP_DENY_LIST),
    corsOrigins: e.CORS_ORIGINS === "*" ? "*" : toArray(e.CORS_ORIGINS),
    docsEnabled: toBoolean(e.DOCS_ENABLED),

    rateLimitEnabled: toBoolean(e.RATE_LIMIT_ENABLED),
    rateLimitMax: e.RATE_LIMIT_PER_WINDOW,
    rateLimitWindowMs: e.RATE_LIMIT_WINDOW_SECONDS * 1000,

    requestTimeoutMs: e.REQUEST_TIMEOUT_SECONDS * 1000,
    maxUploadBytes: e.MAX_UPLOAD_BYTES,
    maxImagePixels: e.MAX_IMAGE_PIXELS,
    maxBatchImages: e.MAX_BATCH_IMAGES,

    executionProviders,
    usesAccelerator,
    defaultEngine: e.DEFAULT_ENGINE,
    modelDir: e.MODEL_DIR,
    math: toBoolean(e.MATH),
    keepFurniture: toBoolean(e.KEEP_FURNITURE),
    pdfDpi: e.PDF_DPI,
    maxPdfPages: e.MAX_PDF_PAGES,

    concurrency: e.MAX_CONCURRENCY > 0 ? e.MAX_CONCURRENCY : autoConcurrency,
    maxQueueDepth: e.MAX_QUEUE_DEPTH,
    taskTtlMs: e.TASK_TTL_SECONDS * 1000,

    sourceUrlAllowlist: toArray(e.SOURCE_URL_ALLOWLIST),
  };
}

export type Config = ReturnType<typeof build>;

export const config: Config = build(process.env);
