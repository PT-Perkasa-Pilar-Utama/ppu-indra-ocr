// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/lm.py (Apache-2.0, Copyright Datalab).
//
// Keys are 64-bit: up to 61 bits of packed tokens. They are kept as a (hi, lo)
// pair of uint32 numbers, so a table lookup is a binary search over the npz
// bytes in place, with no BigInt and no copy. Scores are plain float64 numbers,
// as in the reference.

import { readNpz } from "./npz.js";
import type { NpyArray } from "./npz.js";
import { item } from "./util.js";

export const BITS = 11; // per token, unless the file holds "bits"
export const TAG_SHIFT = 61; // the history length goes in the top 3 bits
export const BOS = -1;

function scalar(z: Map<string, NpyArray>, name: string): number {
  const a = z.get(name);
  const x = a?.data[0];
  if (x === undefined || a?.data.length !== 1) throw new Error(`lm: "${name}" is missing`);
  return Number(x);
}

function table(z: Map<string, NpyArray>, name: string): NpyArray["data"] {
  const a = z.get(name);
  if (!a) throw new Error(`lm: "${name}" is missing`);
  return a.data;
}

/** The uint64 array as interleaved (lo, hi) uint32 pairs, over the same bytes. */
function pairs(a: NpyArray["data"], name: string): Uint32Array {
  if (!(a instanceof BigUint64Array)) throw new Error(`lm: "${name}" is not uint64`);
  return new Uint32Array(a.buffer, a.byteOffset, a.length * 2);
}

function floatTable(a: NpyArray["data"], name: string): Float32Array | Float64Array {
  if (a instanceof Float32Array || a instanceof Float64Array) return a;
  throw new Error(`lm: "${name}" is not a float array`);
}

/** First index in the sorted pair table whose key is >= (hi, lo). */
function lowerBound(keys: Uint32Array, hi: number, lo: number): number {
  let a = 0;
  let b = keys.length >>> 1;
  while (a < b) {
    const mid = (a + b) >>> 1;
    const mh = keys[2 * mid + 1] ?? 0;
    if (mh < hi || (mh === hi && (keys[2 * mid] ?? 0) < lo)) a = mid + 1;
    else b = mid;
  }
  return a;
}

/** Index of the key, or -1. */
function find(keys: Uint32Array, hi: number, lo: number): number {
  const i = lowerBound(keys, hi, lo);
  return i < keys.length >>> 1 && keys[2 * i + 1] === hi && keys[2 * i] === lo ? i : -1;
}

/** Packs tokens (BOS = 0, token t = t + 1) into a key; for tests and tools. */
export function packKey(tokens: readonly number[], bits: number = BITS): bigint {
  let k = 0n;
  for (const t of tokens) k = (k << BigInt(bits)) | BigInt(t === BOS ? 0 : t + 1);
  return k;
}

/**
 * A token n-gram model with backoff: per history length n, sorted keys of the
 * (history, token) pairs with their log-probabilities, and sorted history keys
 * with their backoff weights. Each key array must be sorted and below 2^61.
 */
export class NgramLM {
  readonly order: number;
  readonly vocabSize: number;
  readonly bits: number;
  /** An upper bound of every value logpKey returns. */
  readonly maxLogp: number;

  private readonly k: Uint32Array[] = [];
  private readonly v: (Float32Array | Float64Array)[] = [];
  private readonly hk: Uint32Array[] = [];
  private readonly hb: (Float32Array | Float64Array)[] = [];
  // low (bits * n) bits as (hi, lo), n = 0..order
  private readonly maskHi: number[] = [];
  private readonly maskLo: number[] = [];
  private readonly hist: Int32Array;

  private constructor(z: Map<string, NpyArray>) {
    this.order = scalar(z, "order");
    this.vocabSize = scalar(z, "vocab_size");
    this.bits = z.has("bits") ? scalar(z, "bits") : BITS;
    if (this.order < 1 || this.bits < 1 || this.bits > 31) {
      throw new Error(`lm: unsupported order ${this.order} or bits ${this.bits}`);
    }
    if (this.order * this.bits > TAG_SHIFT) {
      throw new Error(`lm: order ${this.order} x ${this.bits} bits do not fit a key`);
    }
    for (let n = 0; n < this.order; n++) {
      this.k.push(pairs(table(z, `k${n}`), `k${n}`));
      this.v.push(floatTable(table(z, `v${n}`), `v${n}`));
      this.hk.push(pairs(table(z, `hk${n}`), `hk${n}`));
      this.hb.push(floatTable(table(z, `hb${n}`), `hb${n}`));
    }
    for (let n = 0; n <= this.order; n++) {
      const nb = this.bits * n;
      this.maskLo.push(nb >= 32 ? 0xffffffff : ((1 << nb) - 1) >>> 0);
      this.maskHi.push(nb > 32 ? ((1 << (nb - 32)) - 1) >>> 0 : 0);
    }
    this.hist = new Int32Array(this.order);
    let top = -Math.log(this.vocabSize);
    for (const v of this.v) for (let i = 0; i < v.length; i++) top = Math.max(top, v[i] ?? top);
    let bo = 0;
    for (const h of this.hb) {
      if (h.length === 0) continue;
      let m = -Infinity;
      for (let i = 0; i < h.length; i++) m = Math.max(m, h[i] ?? m);
      bo += Math.max(0, m);
    }
    this.maxLogp = top + bo;
  }

  /** Loads a model from the bytes of an lm*.npz file. */
  static async fromNpz(bytes: Uint8Array): Promise<NgramLM> {
    return new NgramLM(await readNpz(bytes));
  }

  /**
   * logp of one (history, token) pair given as one key: the last order - 1
   * history tokens then the token, packed (see packKey), as a (hi, lo) pair of
   * uint32 numbers. Shorter histories are the key's low bits.
   */
  logpPair(hi: number, lo: number): number {
    const b = this.bits;
    const sLo = ((lo >>> b) | (hi << (32 - b))) >>> 0; // key >> bits
    const sHi = hi >>> b;
    const { maskHi, maskLo, hist } = this;
    // m: the longest history length the model has seen
    let m = 0;
    hist[0] = find(item(this.hk, 0), 0, 0);
    for (let n = 1; n < this.order; n++) {
      const j = find(
        item(this.hk, n),
        (sHi & item(maskHi, n)) >>> 0,
        (sLo & item(maskLo, n)) >>> 0
      );
      if (j < 0) break;
      hist[n] = j;
      m = n;
    }
    let acc = 0;
    for (let n = m; n >= 0; n--) {
      const i = find(
        item(this.k, n),
        (hi & item(maskHi, n + 1)) >>> 0,
        (lo & item(maskLo, n + 1)) >>> 0
      );
      if (i >= 0) return acc + item(item(this.v, n), i);
      const j = item(hist, n);
      if (j >= 0) acc += item(item(this.hb, n), j);
    }
    return acc - Math.log(this.vocabSize);
  }

  /** logpPair for keys given as bigint. */
  logpKeys(keys: readonly bigint[]): number[] {
    return keys.map((k) =>
      this.logpPair(Number((k >> 32n) & 0xffffffffn), Number(k & 0xffffffffn))
    );
  }
}

/** log(exp(a) + exp(b)) with -inf as zero probability. */
function lse(a: number, b: number): number {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

function compare(a: number, b: number): number {
  if (a < b) return -1;
  return a > b ? 1 : 0;
}

type Node = { id: number; step: number; parent: Node | null };

type Beam = {
  b: number; // log p of the prefix ending in blank
  nb: number; // log p ending in a non-blank
  s: number; // LM score
  node: Node | null; // the prefix's last token
  hHi: number; // the prefix's last order - 1 tokens, packed
  hLo: number;
};

type Ext = { key: string; c: number; newFrom: number; beam: Beam };

export type BeamResult = {
  /** The token ids. */
  ids: number[];
  /** The step where each token was first emitted. */
  steps: number[];
};

/**
 * CTC prefix beam search over a line's T steps, scored as
 * log P_ctc + alpha * log P_lm + beta * tokens.
 *
 * blank: [T] blank log-probabilities. topv, topi: [T * k] the k best classes
 * per step (blank included) and their log-probabilities, row-major. topi takes
 * numbers or bigints, so an int64 tensor's BigInt64Array works as it is. A
 * class under `candFloor` probability at a step is not tried there.
 */
export function beamSearch(
  blank: ArrayLike<number>,
  topv: ArrayLike<number>,
  topi: ArrayLike<number | bigint>,
  T: number,
  k: number,
  lm: NgramLM,
  alpha = 0.5,
  beta = 1.5,
  beam = 8,
  blankSkip = 0.999,
  candFloor = 1e-3
): BeamResult {
  const bits = lm.bits;
  const hmaskBits = bits * (lm.order - 1);
  const hmaskLo = hmaskBits >= 32 ? 0xffffffff : ((1 << hmaskBits) - 1) >>> 0;
  const hmaskHi = hmaskBits > 32 ? ((1 << (hmaskBits - 32)) - 1) >>> 0 : 0;
  let beams = new Map<string, Beam>([
    ["", { b: 0, nb: -Infinity, s: 0, node: null, hHi: 0, hLo: 0 }],
  ]);
  const logSkip = Math.log(blankSkip);
  const logFloor = Math.log(candFloor);
  const reach = beta + alpha * lm.maxLogp;
  for (let t = 0; t < T; t++) {
    const b0 = item(blank, t);
    if (b0 > logSkip) {
      const skipped = new Map<string, Beam>();
      for (const [p, e] of beams) skipped.set(p, { ...e, b: lse(e.b, e.nb) + b0, nb: -Infinity });
      beams = skipped;
      continue;
    }
    const nxt = new Map<string, Beam>();
    const add = (key: string, b: number, nb: number, s: number, node: Node | null, h: Beam) => {
      const old = nxt.get(key);
      if (old === undefined) nxt.set(key, { b, nb, s, node, hHi: h.hHi, hLo: h.hLo });
      else {
        old.b = lse(old.b, b);
        old.nb = lse(old.nb, nb);
      }
    };
    const cands: { c: number; lp: number }[] = [];
    for (let j = 0; j < k; j++) {
      const c = Number(item(topi, t * k + j));
      const v = item(topv, t * k + j);
      if (c !== 0 && v >= logFloor) {
        cands.push({ c, lp: v });
      }
    }
    // Every beam's own entries first; see lm.py for why they set a floor.
    for (const [key, e] of beams) {
      add(key, lse(e.b, e.nb) + b0, -Infinity, e.s, e.node, e);
      const last = e.node ? e.node.id : -1;
      for (const { c, lp } of cands) {
        if (c === last) add(key, -Infinity, e.nb + lp, e.s, e.node, e);
      }
    }
    let floor = -Infinity;
    if (beams.size >= beam) {
      floor = Infinity;
      for (const e of nxt.values()) floor = Math.min(floor, lse(e.b, e.nb) + e.s);
    }
    const ext: Ext[] = [];
    for (const [key, e] of beams) {
      const tot = lse(e.b, e.nb);
      const last = e.node ? e.node.id : -1;
      for (const { c, lp } of cands) {
        // after a blank, the same class is a new token
        const newFrom = c === last ? e.b + lp : tot + lp;
        if (newFrom === -Infinity) continue;
        const q = `${key}${c},`;
        if (nxt.has(q)) add(q, -Infinity, newFrom, e.s, e.node, e);
        else if (newFrom + e.s + reach >= floor - 1e-9) ext.push({ key: q, c, newFrom, beam: e });
      }
    }
    for (const { key, c, newFrom, beam: e } of ext) {
      // (history << bits) | (c + 1)
      const hi = ((e.hHi << bits) | (e.hLo >>> (32 - bits))) >>> 0;
      const lo = ((e.hLo << bits) | (c + 1)) >>> 0;
      const lmp = lm.logpPair(hi, lo);
      const node: Node = { id: c, step: t, parent: e.node };
      add(key, -Infinity, newFrom, e.s + alpha * lmp + beta, node, {
        ...e,
        hHi: (hi & hmaskHi) >>> 0,
        hLo: (lo & hmaskLo) >>> 0,
      });
    }
    const ranked = [...nxt].map((kv) => ({ kv, score: -(lse(kv[1].b, kv[1].nb) + kv[1].s) }));
    ranked.sort((x, y) => compare(x.score, y.score));
    beams = new Map(ranked.slice(0, beam).map((r) => r.kv));
  }
  let best: Beam | undefined;
  let bestScore = -Infinity;
  for (const e of beams.values()) {
    const score = lse(e.b, e.nb) + e.s;
    if (best === undefined || score > bestScore) {
      best = e;
      bestScore = score;
    }
  }
  const ids: number[] = [];
  const steps: number[] = [];
  for (let n = best ? best.node : null; n; n = n.parent) {
    ids.push(n.id);
    steps.push(n.step);
  }
  return { ids: ids.reverse(), steps: steps.reverse() };
}
