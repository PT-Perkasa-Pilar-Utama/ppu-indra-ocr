import { describe, expect, test } from "bun:test";

import { sumF32 } from "../../src/indra/numeric.js";

// numpy 2.5.4: np.array([((i * 7919) % 1009) / 7.3 - 60 for i in range(n)]).astype(np.float32).sum(dtype=np.float32)
const NUMPY: [n: number, sum: number][] = [
  [1, -60.0],
  [7, -30.821924209594727],
  [8, 38.90410614013672],
  [9, 87.67123413085938],
  [127, 1193.69873046875],
  [128, 1236.3013916015625],
  [129, 1257.9453125],
  [450, 4105.6162109375],
  [1000, 9133.97265625],
  [4097, 37046.578125],
];

const values = (n: number): Float32Array =>
  Float32Array.from({ length: n }, (_, i) => ((i * 7919) % 1009) / 7.3 - 60);

describe("sumF32", () => {
  test("matches numpy's float32 sum bit for bit, across the block edges", () => {
    for (const [n, want] of NUMPY) expect([n, sumF32(values(n))]).toEqual([n, want]);
  });

  test("sums a slice like numpy sums that slice", () => {
    const a = values(1000);
    expect(sumF32(a, 550, 1000)).toBe(sumF32(a.slice(550)));
  });
});
