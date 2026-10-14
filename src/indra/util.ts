// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import type { InferenceSession, Tensor } from "onnxruntime-common";

/** `arr[i]`, or a RangeError: for indices the algorithm guarantees are in range. */
export function item<T>(arr: ArrayLike<T>, i: number): T {
  const v = arr[i];
  if (v === undefined) throw new RangeError(`index ${i} out of range (length ${arr.length})`);
  return v;
}

/** A named model output, or an error naming the missing output. */
export function output(res: InferenceSession.OnnxValueMapType, name: string): Tensor {
  const t = res[name];
  if (!t) throw new Error(`model output "${name}" is missing`);
  return t;
}

/** A float32 tensor's data. */
export function floats(t: Tensor): Float32Array {
  if (t.type !== "float32") throw new Error(`expected a float32 tensor, got ${t.type}`);
  // SAFETY: onnxruntime backs float32 tensors with a Float32Array (type checked above).
  return t.data as Float32Array;
}

/** An integer tensor's data as numbers (int64 ids and lengths fit in 2^53). */
export function ints(t: Tensor): number[] {
  if (t.type !== "int64" && t.type !== "int32") {
    throw new Error(`expected an integer tensor, got ${t.type}`);
  }
  // SAFETY: int64 tensors hold a BigInt64Array, int32 an Int32Array (type checked above).
  return Array.from(t.data as ArrayLike<bigint | number>, Number);
}
