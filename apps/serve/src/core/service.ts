import { join } from "node:path";

import { IndraOcrService, MODEL_FILES } from "ppu-indra-ocr";
import type { ModelSources, ProcessingEngine } from "ppu-indra-ocr";
import { config } from "./config.js";
import { InferenceQueue } from "./queue.js";

/** Shared inference gate across every engine/service (one device, one budget). */
export const queue = new InferenceQueue(config.concurrency, config.maxQueueDepth);

// Up to two warmed services (one per engine), created lazily. The default
// engine is warmed at boot; the other is built on first request that asks for
// it. Both share the inference queue.
const services = new Map<ProcessingEngine, IndraOcrService>();
let ready = false;

/** Every model file from MODEL_DIR, or undefined to use the cached default download. */
function models(): ModelSources | undefined {
  const dir = config.modelDir;
  if (!dir) return undefined;
  return {
    det: join(dir, MODEL_FILES.det),
    rec: join(dir, MODEL_FILES.rec),
    recCjk: join(dir, MODEL_FILES.recCjk),
    script: join(dir, MODEL_FILES.script),
    orient: join(dir, MODEL_FILES.orient),
    order: join(dir, MODEL_FILES.order),
    meta: join(dir, MODEL_FILES.meta),
    mathCls: join(dir, MODEL_FILES.mathCls),
    mathEnc: join(dir, MODEL_FILES.mathEnc),
    mathDecInit: join(dir, MODEL_FILES.mathDecInit),
    mathDecStep: join(dir, MODEL_FILES.mathDecStep),
    mathMeta: join(dir, MODEL_FILES.mathMeta),
  };
}

function build(engine: ProcessingEngine): IndraOcrService {
  return new IndraOcrService({
    session: { executionProviders: config.executionProviders },
    engine,
    math: config.math,
    furniture: config.keepFurniture,
    models: models(),
  });
}

/** Warm the default-engine service. Call once at boot before serving. */
export async function initService(): Promise<void> {
  const svc = build(config.defaultEngine);
  await svc.initialize();
  services.set(config.defaultEngine, svc);
  ready = true;
}

/** Get (or lazily warm) the service for an engine. */
export async function getService(
  engine: ProcessingEngine = config.defaultEngine
): Promise<IndraOcrService> {
  const existing = services.get(engine);
  if (existing) return existing;
  const svc = build(engine);
  await svc.initialize();
  services.set(engine, svc);
  return svc;
}

/** Readiness: true once the default engine is warmed. */
export function isReady(): boolean {
  return ready;
}

/** Release every warmed service. Call on shutdown. */
export async function shutdownService(): Promise<void> {
  ready = false;
  for (const svc of services.values()) await svc.destroy();
  services.clear();
}
