// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import type { InferenceSession } from "onnxruntime-common";

import type { DetBox } from "../indra/det-post.js";
import type { Gray } from "../indra/image.js";
import { grayFromRgba } from "../indra/image.js";
import type { MathMeta, MathReading } from "../indra/math.js";
import { MathReader } from "../indra/math.js";
import { IndraPipeline } from "../indra/pipeline.js";
import type { Box } from "../indra/rec-preproc.js";
import { tsImageOps } from "../indra/ts-image-ops.js";
import type {
  IndraMeta,
  Orientation,
  Page,
  ReadingOrder,
  RecLine,
  Straightened,
} from "../indra/types.js";
import type { BatchOptions, IndraOptions, ProcessingEngine } from "../interface.js";
import type { ModelFile } from "../model-catalogue.js";
import { BASE_FILES, MATH_FILES, MODEL_FILES, defaultModelUrl } from "../model-catalogue.js";
import type { BatchItemResult, RunPoolOptions } from "./batch.js";
import { createAsyncQueue, runPool } from "./batch.js";
import { opencvImageOps } from "./opencv-image-ops.js";
import type { CoreCanvas, PlatformProvider } from "./platform.js";
import { createSessionWithFallback } from "./session-factory.js";

/** An image to read: encoded bytes (PNG, JPEG, ...), a platform canvas, or 8-bit gray pixels. */
export type ImageInput<TCanvas = CoreCanvas> = ArrayBuffer | Uint8Array | TCanvas | Gray;

function isGray<T>(image: ImageInput<T>): image is Gray {
  return (
    typeof image === "object" &&
    image !== null &&
    "data" in image &&
    image.data instanceof Uint8Array &&
    "width" in image &&
    "height" in image &&
    image.data.length === Number(image.width) * Number(image.height)
  );
}

/**
 * Indra OCR on one platform. Call `initialize()` once, then `convert()` a page,
 * or run the stages yourself: `orient`, `straighten`, `detect`, `recognize`,
 * `order`, `readMath`. Call `destroy()` to release the models.
 */
export abstract class BaseIndraOcrService<TCanvas = CoreCanvas> {
  protected readonly options: IndraOptions;
  private indra: IndraPipeline | null = null;
  private sessions: InferenceSession[] = [];
  private engineInUse: ProcessingEngine | null = null;

  protected constructor(
    protected readonly platform: PlatformProvider<TCanvas>,
    options: IndraOptions = {}
  ) {
    this.options = { ...options };
  }

  protected log(message: string): void {
    if (this.options.verbose) console.log(`[IndraOcrService] ${message}`);
  }

  public isInitialized(): boolean {
    return this.indra !== null;
  }

  /** The engine in use after initialize(). */
  public get engine(): ProcessingEngine | null {
    return this.engineInUse;
  }

  /** The stage pipeline. Throws before initialize(). */
  public get pipeline(): IndraPipeline {
    if (!this.indra) throw new Error("IndraOcrService is not initialized: call initialize() first");
    return this.indra;
  }

  /** Load the models (and OpenCV for the opencv engine). Safe to call twice. */
  public async initialize(): Promise<void> {
    if (this.indra) return;
    const roles = this.options.math ? [...BASE_FILES, ...MATH_FILES] : BASE_FILES;
    const bytes = new Map(
      await Promise.all(
        roles.map(async (role): Promise<[ModelFile, ArrayBuffer]> => [role, await this.load(role)])
      )
    );
    const text = (role: ModelFile): string => new TextDecoder().decode(bytes.get(role));
    const session = async (role: ModelFile): Promise<InferenceSession> => {
      const s = await createSessionWithFallback(
        this.platform.ort,
        new Uint8Array(bytes.get(role) ?? new ArrayBuffer(0)),
        this.options.session,
        (msg) => this.log(msg),
        (next) => (this.options.session = next)
      );
      this.sessions.push(s);
      return s;
    };
    const ops = await this.imageOps();
    // SAFETY: meta.json and math_meta.json ship with the models in these shapes.
    const meta = JSON.parse(text("meta")) as IndraMeta;
    const math = this.options.math
      ? new MathReader(
          this.platform.ort.Tensor,
          {
            cls: await session("mathCls"),
            enc: await session("mathEnc"),
            init: await session("mathDecInit"),
            step: await session("mathDecStep"),
          },
          // SAFETY: as above.
          JSON.parse(text("mathMeta")) as MathMeta,
          ops
        )
      : undefined;
    this.indra = new IndraPipeline(
      this.platform.ort.Tensor,
      {
        det: await session("det"),
        rec: await session("rec"),
        recCjk: await session("recCjk"),
        script: await session("script"),
        orient: await session("orient"),
        order: await session("order"),
      },
      meta,
      ops,
      { furniture: this.options.furniture, adaptive: this.options.adaptive, math }
    );
    this.log(`ready: ${roles.length} model files, engine ${this.engineInUse}`);
  }

  private async load(role: ModelFile): Promise<ArrayBuffer> {
    const source = this.options.models?.[role];
    try {
      return await this.platform.loadResource(source, defaultModelUrl(role));
    } catch (err) {
      const from =
        source instanceof ArrayBuffer ? "the given bytes" : (source ?? defaultModelUrl(role));
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`could not load ${MODEL_FILES[role]} from ${from}: ${msg}`, { cause: err });
    }
  }

  private async imageOps(): Promise<typeof tsImageOps> {
    const opencv = this.platform.opencv;
    const engine = this.options.engine ?? this.platform.defaultEngine;
    this.engineInUse = engine;
    if (engine === "canvas-native") return tsImageOps;
    if (!opencv) throw new Error('the "opencv" engine is not available on this platform');
    return opencvImageOps(await opencv());
  }

  /** Release the models. The service can be initialized again. */
  public async destroy(): Promise<void> {
    await Promise.all(this.sessions.map((s) => s.release()));
    this.sessions = [];
    this.indra = null;
  }

  /** 8-bit gray pixels of an image (OpenCV's BT.601 weights for color). */
  public async toGray(image: ImageInput<TCanvas>): Promise<Gray> {
    if (isGray(image)) return image;
    if (this.platform.isCanvas(image)) return grayFromRgba(this.platform.pixels(image));
    if (image instanceof ArrayBuffer || image instanceof Uint8Array) {
      const buf =
        image instanceof Uint8Array
          ? image.buffer.slice(image.byteOffset, image.byteOffset + image.byteLength)
          : image;
      // SAFETY: slice() of a Uint8Array's buffer is an ArrayBuffer, never shared.
      const canvas = await this.platform.decodeImage(buf as ArrayBuffer);
      return grayFromRgba(this.platform.pixels(canvas));
    }
    throw new TypeError(
      "image must be encoded bytes, a canvas, or { data, width, height } gray pixels"
    );
  }

  /** OCR of one page image: markdown, text, lines and layout blocks. */
  public async convert(image: ImageInput<TCanvas>): Promise<Page> {
    return this.pipeline.page(await this.toGray(image));
  }

  /** Convert many images. Results keep the input order. */
  public async convertBatch(
    images: Iterable<ImageInput<TCanvas>> | AsyncIterable<ImageInput<TCanvas>>,
    options: BatchOptions & { settle: true }
  ): Promise<BatchItemResult<Page>[]>;
  public async convertBatch(
    images: Iterable<ImageInput<TCanvas>> | AsyncIterable<ImageInput<TCanvas>>,
    options?: BatchOptions
  ): Promise<Page[]>;
  public async convertBatch(
    images: Iterable<ImageInput<TCanvas>> | AsyncIterable<ImageInput<TCanvas>>,
    options: BatchOptions = {}
  ): Promise<Page[] | BatchItemResult<Page>[]> {
    const collected: BatchItemResult<Page>[] = [];
    await runPool(
      images,
      this.poolOptions(images, options),
      (image) => this.convert(image),
      (r) => {
        collected[r.index] = r;
      }
    );
    if (options.settle) return collected;
    // Without settle, runPool rethrows the first failure, so every item here is fulfilled.
    return collected.flatMap((r) => (r.status === "fulfilled" ? [r.value] : []));
  }

  /** Convert many images, yielding each result as it finishes (completion order). */
  public async *convertStream(
    images: Iterable<ImageInput<TCanvas>> | AsyncIterable<ImageInput<TCanvas>>,
    options: BatchOptions = {}
  ): AsyncGenerator<BatchItemResult<Page>> {
    const queue = createAsyncQueue<BatchItemResult<Page>>();
    const pump = (async (): Promise<void> => {
      try {
        await runPool(
          images,
          this.poolOptions(images, options),
          (image) => this.convert(image),
          (r) => queue.push(r)
        );
        queue.close();
      } catch (error) {
        queue.fail(error);
      }
    })();
    yield* queue.drain();
    await pump;
  }

  private poolOptions(images: unknown, options: BatchOptions): RunPoolOptions {
    return {
      concurrency: options.concurrency ?? 1,
      settle: options.settle ?? false,
      signal: options.signal,
      onProgress: options.onProgress,
      total: Array.isArray(images) ? images.length : undefined,
    };
  }

  /** Quarter turns (counter-clockwise) the content is turned by, from orient.onnx. */
  public async orient(image: ImageInput<TCanvas>): Promise<Orientation> {
    return this.pipeline.orient(await this.toGray(image));
  }

  /** The page turned upright by `turns` quarter turns and deskewed. */
  public async straighten(image: ImageInput<TCanvas>, turns = 0): Promise<Straightened> {
    return this.pipeline.straighten(await this.toGray(image), turns);
  }

  /** Line boxes (x0, y0, x1, y1, score) in image pixels. */
  public async detect(image: ImageInput<TCanvas>): Promise<DetBox[]> {
    return this.pipeline.detect(await this.toGray(image));
  }

  /**
   * Text, confidence and character positions of each box, one line per box.
   * Boxes should hug the text: the recognizer adds 0.2 x the box height above
   * and below and 0.3 x on each side. A box from another detector that carries
   * its own padding shrinks the text and reads worse; trim it first.
   */
  public async recognize(image: ImageInput<TCanvas>, boxes: readonly Box[]): Promise<RecLine[]> {
    return this.pipeline.recognize(await this.toGray(image), boxes);
  }

  /** Reading order of lines, with the layout heads' class and same-block logits. */
  public async order(
    boxes: readonly Box[],
    texts: readonly string[],
    width: number,
    height: number
  ): Promise<ReadingOrder> {
    return this.pipeline.readingOrder(boxes, texts, width, height);
  }

  /** Math readings (text with inline LaTeX, mean token probability) of the boxes. Needs `math: true`. */
  public async readMath(image: ImageInput<TCanvas>, boxes: readonly Box[]): Promise<MathReading[]> {
    const math = this.pipeline.mathReader;
    if (!math) throw new Error("readMath needs the service created with { math: true }");
    return math.read(await this.toGray(image), boxes);
  }
}
