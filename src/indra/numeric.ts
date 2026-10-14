// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

const f = Math.fround;

/**
 * Sum of float32 values the way numpy sums a contiguous float32 array:
 * pairwise, in blocks of up to 128 with 8 lanes, rounding to float32 at every
 * step. The reference compares sums (skew search) and thresholds means
 * (confidence), so a plain double sum would flip some of its decisions.
 */
export function sumF32(a: ArrayLike<number>, start = 0, end: number = a.length): number {
  const n = end - start;
  if (n < 8) {
    let r = 0;
    for (let i = start; i < end; i++) r = f(r + (a[i] ?? 0));
    return r;
  }
  if (n <= 128) {
    const r = [0, 0, 0, 0, 0, 0, 0, 0];
    for (let j = 0; j < 8; j++) r[j] = f(a[start + j] ?? 0);
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (let j = 0; j < 8; j++) r[j] = f((r[j] ?? 0) + (a[start + i + j] ?? 0));
    }
    const [r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0] = r;
    let res = f(f(f(r0 + r1) + f(r2 + r3)) + f(f(r4 + r5) + f(r6 + r7)));
    for (; i < n; i++) res = f(res + (a[start + i] ?? 0));
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return f(sumF32(a, start, start + n2) + sumF32(a, start + n2, end));
}

/** numpy's float32 mean: the pairwise sum divided in float32. */
export function meanF32(a: ArrayLike<number>): number {
  return f(sumF32(a) / a.length);
}
