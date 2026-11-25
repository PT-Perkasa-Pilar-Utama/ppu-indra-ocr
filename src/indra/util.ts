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

/**
 * The shape of a model input as the session reports it (symbolic dimensions
 * as strings), or undefined for an input it does not have or that is not a
 * tensor. Some runtimes have no input metadata (onnxruntime-react-native
 * before 1.24.1, the WebGL backend): those throw, with the runtime's error as
 * the cause, since no default size is right for every model.
 */
export function inputShape(
  sess: InferenceSession,
  name: string
): readonly (number | string)[] | undefined {
  let meta: readonly InferenceSession.ValueMetadata[];
  try {
    meta = sess.inputMetadata;
  } catch (err) {
    throw new Error(
      `this ONNX Runtime gives no input metadata, which the "${name}" input needs ` +
        "(onnxruntime-react-native 1.24.1 or later, or a backend other than WebGL)",
      { cause: err }
    );
  }
  const input = meta.find((m) => m.name === name);
  return input?.isTensor ? input.shape : undefined;
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

/**
 * Python's str.isspace for one character: JS \s, plus U+001C..U+001F and U+0085,
 * minus U+FEFF (which JS counts as whitespace and Python does not).
 */
export function isPySpace(c: string): boolean {
  const o = c.codePointAt(0) ?? 0;
  if (o === 0xfeff) return false;
  return (o >= 0x1c && o <= 0x1f) || o === 0x85 || /^\s$/u.test(c);
}
