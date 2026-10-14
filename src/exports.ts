// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

// Types and pure helpers shared by the node, web and mobile entries.

export type { ImageInput } from "./core/indra-ocr.service.js";
export type { DetBox } from "./indra/det-post.js";
export type { Gray } from "./indra/image.js";
export type { Cell, LayoutBlock } from "./indra/layout-post.js";
export { joinLines, pageMarkdown as toMarkdown } from "./indra/layout-post.js";
export type { MathReading } from "./indra/math.js";
export { BLOCK_CLASSES, orderXYCut } from "./indra/order-post.js";
export type { BlockLabel } from "./indra/order-post.js";
export type { Box } from "./indra/rec-preproc.js";
export type {
  Line,
  Orientation,
  Page,
  PageSource,
  ReadingOrder,
  RecLine,
  Straightened,
  Tok,
} from "./indra/types.js";
export type { IndraOptions, ModelSources, ProcessingEngine, SessionOptions } from "./interface.js";
export {
  MATH_FILES,
  MODEL_BASE_URL,
  MODEL_FILES,
  MODEL_MIRROR_BASE_URL,
  MODEL_REVISION,
  defaultModelUrl,
} from "./model-catalogue.js";
export type { ModelFile } from "./model-catalogue.js";
