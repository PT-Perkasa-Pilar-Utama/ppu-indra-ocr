// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import type { InferenceSession } from "onnxruntime-common";

import type { ImageOps } from "../indra/image.js";
import type { SpanImgMeta } from "../indra/math-span-img.js";
import { SpanImgTagger } from "../indra/math-span-img.js";
import type { MathAcceptHead, MathGateHead } from "../indra/math-text.js";
import type { MathMeta } from "../indra/math.js";
import { MathReader } from "../indra/math.js";
import type { TensorCtor } from "../indra/types.js";
import type { ModelFile } from "../model-catalogue.js";

/**
 * The math reader of a model set: its decoder step graph (the per-layer one
 * when the set has it), and the accept head, read gate and inline formula
 * tagger it has. `text` reads a JSON file the caller already loaded.
 */
export async function loadMathReader(
  Tensor: TensorCtor,
  has: (role: ModelFile) => boolean,
  session: (role: ModelFile) => Promise<InferenceSession>,
  text: (role: ModelFile) => string,
  ops: ImageOps
): Promise<MathReader> {
  // SAFETY: the math JSON files ship with the models in these shapes.
  const json = <T>(role: ModelFile): T => JSON.parse(text(role)) as T;
  const spanImg =
    has("mathSpanImg") && has("mathSpanImgMeta")
      ? new SpanImgTagger(
          Tensor,
          await session("mathSpanImg"),
          json<SpanImgMeta>("mathSpanImgMeta")
        )
      : undefined;
  return new MathReader(
    Tensor,
    {
      cls: await session("mathCls"),
      enc: await session("mathEnc"),
      init: await session("mathDecInit"),
      step: await session(has("mathDecStepKv") ? "mathDecStepKv" : "mathDecStep"),
    },
    json<MathMeta>("mathMeta"),
    ops,
    {
      ...(has("mathAccept") ? { accept: json<MathAcceptHead>("mathAccept") } : {}),
      ...(has("mathGate") ? { gate: json<MathGateHead>("mathGate") } : {}),
      ...(spanImg ? { spanImg } : {}),
    }
  );
}
