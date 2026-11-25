// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { InferenceSession, Tensor } from "onnxruntime-common";

import { pyRound } from "./det-post.js";
import type { Gray, ImageOps, Interpolation } from "./image.js";
import type { Box } from "./rec-preproc.js";
import type { TensorCtor } from "./types.js";
import { inputShape } from "./util.js";

/** order.onnx line thumbnail size when the model does not state it. */
const THUMB_H = 16;
const THUMB_W = 64;
/** Line heights of context above and below each line box. */
const THUMB_CTX = 0.25;

/** The thumbnail size and resize mode an order model reads. */
export type ThumbSpec = { height: number; width: number; interp: Interpolation };

/**
 * The thumbnail spec of an order model that reads line thumbnails (an input
 * named "thumbs"), or undefined for one that does not. The size comes from
 * the input's shape, the resize mode from meta.json (default area).
 */
export function thumbSpec(
  order: InferenceSession,
  interp: Interpolation | undefined
): ThumbSpec | undefined {
  if (!order.inputNames.includes("thumbs")) return undefined;
  const shape = inputShape(order, "thumbs") ?? [];
  const h = shape.at(-2);
  const w = shape.at(-1);
  return typeof h === "number" && typeof w === "number"
    ? { height: h, width: w, interp: interp ?? "area" }
    : { height: THUMB_H, width: THUMB_W, interp: interp ?? "area" };
}

/**
 * [N, 1, h, w] float32 thumbnails of line boxes (pixels of `gray`): the box
 * with THUMB_CTX line heights above and below, scaled to h rows with its
 * aspect kept (cut at w), ink = 1. The order model expects exactly these.
 */
export function lineThumbs(
  gray: Gray,
  boxes: readonly Box[],
  { height: th, width: tw, interp }: ThumbSpec,
  ops: ImageOps
): Float32Array {
  const out = new Float32Array(boxes.length * th * tw);
  const { width: W, height: H } = gray;
  for (const [i, [x0, y0, x1, y1]] of boxes.entries()) {
    const h = Math.max(2, y1 - y0);
    const a = Math.trunc(Math.max(0, y0 - THUMB_CTX * h));
    const z = Math.trunc(Math.min(H, y1 + THUMB_CTX * h));
    const c = Math.trunc(Math.max(0, x0));
    const d = Math.trunc(Math.min(W, x1));
    if (z - a < 2 || d - c < 1) continue;
    const crop = new Uint8Array((z - a) * (d - c));
    for (let y = a; y < z; y++) {
      crop.set(gray.data.subarray(y * W + c, y * W + d), (y - a) * (d - c));
    }
    const w = Math.max(1, Math.min(tw, pyRound(((d - c) * th) / (z - a))));
    const r = ops.resize({ data: crop, width: d - c, height: z - a }, w, th, interp);
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < w; x++) {
        out[(i * th + y) * tw + x] = (255 - (r.data[y * w + x] ?? 0)) / 255;
      }
    }
  }
  return out;
}

/** The order model's thumbnail inputs. */
export type ThumbFeed = { thumbs: Tensor; has_img: Tensor };

/**
 * The line thumbnails of an order model that reads them, from the page image
 * at `imgBoxes` (the boxes on the page as it is; the order model's boxes are
 * mirrored on a right-to-left page); without the image the lines have none
 * (has_img 0).
 */
export function thumbFeed(
  Tensor: TensorCtor,
  ops: ImageOps,
  spec: ThumbSpec,
  n: number,
  gray: Gray | undefined,
  imgBoxes: readonly Box[]
): ThumbFeed {
  const { height: th, width: tw } = spec;
  const thumbs = gray ? lineThumbs(gray, imgBoxes, spec, ops) : new Float32Array(n * th * tw);
  return {
    thumbs: new Tensor("float32", thumbs, [1, n, 1, th, tw]),
    has_img: new Tensor("float32", new Float32Array(n).fill(gray ? 1 : 0), [1, n]),
  };
}
