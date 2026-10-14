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
    // Eight named lanes, not an array: this leaf runs for every 128 values of every skew candidate.
    let r0 = f(a[start] ?? 0);
    let r1 = f(a[start + 1] ?? 0);
    let r2 = f(a[start + 2] ?? 0);
    let r3 = f(a[start + 3] ?? 0);
    let r4 = f(a[start + 4] ?? 0);
    let r5 = f(a[start + 5] ?? 0);
    let r6 = f(a[start + 6] ?? 0);
    let r7 = f(a[start + 7] ?? 0);
    const stop = start + n - (n % 8);
    let i = start + 8;
    for (; i < stop; i += 8) {
      r0 = f(r0 + (a[i] ?? 0));
      r1 = f(r1 + (a[i + 1] ?? 0));
      r2 = f(r2 + (a[i + 2] ?? 0));
      r3 = f(r3 + (a[i + 3] ?? 0));
      r4 = f(r4 + (a[i + 4] ?? 0));
      r5 = f(r5 + (a[i + 5] ?? 0));
      r6 = f(r6 + (a[i + 6] ?? 0));
      r7 = f(r7 + (a[i + 7] ?? 0));
    }
    let res = f(f(f(r0 + r1) + f(r2 + r3)) + f(f(r4 + r5) + f(r6 + r7)));
    for (; i < end; i++) res = f(res + (a[i] ?? 0));
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
