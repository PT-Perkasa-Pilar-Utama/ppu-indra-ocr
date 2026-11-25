// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from CPython Lib/difflib.py 3.12 (Python Software Foundation License 2.0).
//
// difflib.SequenceMatcher(None, a, b, autojunk=False): no junk function and no
// popular-element rule, so b2j holds every element of b. Ties break as CPython
// breaks them: the longest block that starts first in a, then first in b.

import { item } from "./util.js";

/** A block a[a:a+size] == b[b:b+size] (CPython's Match). */
export type Match = { readonly a: number; readonly b: number; readonly size: number };

export type OpcodeTag = "replace" | "delete" | "insert" | "equal";

/** (tag, i1, i2, j1, j2): CPython's opcode tuple. */
export type Opcode = readonly [tag: OpcodeTag, i1: number, i2: number, j1: number, j2: number];

/** A string is a sequence of its code points, as in Python. */
export type Seq = string | readonly string[];

const elements = (s: Seq): readonly string[] => (typeof s === "string" ? Array.from(s) : s);

export class SequenceMatcher {
  private readonly a: readonly string[];
  private readonly b: readonly string[];
  private readonly b2j = new Map<string, number[]>();
  private blocks: Match[] | undefined;
  private ops: Opcode[] | undefined;

  constructor(a: Seq, b: Seq) {
    this.a = elements(a);
    this.b = elements(b);
    this.b.forEach((elt, j) => {
      const indices = this.b2j.get(elt);
      if (indices) indices.push(j);
      else this.b2j.set(elt, [j]);
    });
  }

  /**
   * The longest matching block in a[alo:ahi] and b[blo:bhi]; of the longest, the
   * one that starts first in a, and of those the one that starts first in b.
   * With no match, (alo, blo, 0).
   */
  findLongestMatch(
    alo: number = 0,
    ahi: number = this.a.length,
    blo: number = 0,
    bhi: number = this.b.length
  ): Match {
    const { a, b } = this;
    let besti = alo;
    let bestj = blo;
    let bestsize = 0;
    // j2len.get(j) is the length of the longest match that ends with a[i - 1] and b[j]
    let j2len = new Map<number, number>();
    for (let i = alo; i < ahi; i++) {
      const newj2len = new Map<number, number>();
      for (const j of this.b2j.get(item(a, i)) ?? []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (j2len.get(j - 1) ?? 0) + 1;
        newj2len.set(j, k);
        if (k > bestsize) {
          besti = i - k + 1;
          bestj = j - k + 1;
          bestsize = k;
        }
      }
      j2len = newj2len;
    }
    // CPython also extends the match over junk and popular elements; with neither
    // in play these loops end at once, and they stay so the port reads like the source.
    while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) {
      besti--;
      bestj--;
      bestsize++;
    }
    while (
      besti + bestsize < ahi &&
      bestj + bestsize < bhi &&
      a[besti + bestsize] === b[bestj + bestsize]
    ) {
      bestsize++;
    }
    return { a: besti, b: bestj, size: bestsize };
  }

  /**
   * Maximal non-overlapping matching blocks, increasing in a and in b, with no two
   * adjacent; the last one is the sentinel (len(a), len(b), 0).
   */
  getMatchingBlocks(): readonly Match[] {
    if (this.blocks) return this.blocks;
    const la = this.a.length;
    const lb = this.b.length;
    const queue: [number, number, number, number][] = [[0, la, 0, lb]];
    const found: Match[] = [];
    for (let next = queue.pop(); next; next = queue.pop()) {
      const [alo, ahi, blo, bhi] = next;
      const m = this.findLongestMatch(alo, ahi, blo, bhi);
      if (m.size === 0) continue;
      found.push(m);
      if (alo < m.a && blo < m.b) queue.push([alo, m.a, blo, m.b]);
      if (m.a + m.size < ahi && m.b + m.size < bhi) {
        queue.push([m.a + m.size, ahi, m.b + m.size, bhi]);
      }
    }
    found.sort((x, y) => x.a - y.a || x.b - y.b || x.size - y.size);
    // collapse blocks that touch
    const blocks: Match[] = [];
    let cur: Match = { a: 0, b: 0, size: 0 };
    for (const m of found) {
      if (cur.a + cur.size === m.a && cur.b + cur.size === m.b) {
        cur = { a: cur.a, b: cur.b, size: cur.size + m.size };
      } else {
        if (cur.size) blocks.push(cur);
        cur = m;
      }
    }
    if (cur.size) blocks.push(cur);
    blocks.push({ a: la, b: lb, size: 0 });
    this.blocks = blocks;
    return blocks;
  }

  /** The edit script that turns a into b. */
  getOpcodes(): readonly Opcode[] {
    if (this.ops) return this.ops;
    let i = 0;
    let j = 0;
    const ops: Opcode[] = [];
    for (const { a: ai, b: bj, size } of this.getMatchingBlocks()) {
      if (i < ai && j < bj) ops.push(["replace", i, ai, j, bj]);
      else if (i < ai) ops.push(["delete", i, ai, j, bj]);
      else if (j < bj) ops.push(["insert", i, ai, j, bj]);
      i = ai + size;
      j = bj + size;
      if (size) ops.push(["equal", ai, i, bj, j]);
    }
    this.ops = ops;
    return ops;
  }

  /** 2.0 * matches / (len(a) + len(b)), or 1.0 for two empty sequences. */
  ratio(): number {
    const matches = this.getMatchingBlocks().reduce((n, m) => n + m.size, 0);
    const length = this.a.length + this.b.length;
    return length ? (2.0 * matches) / length : 1.0;
  }
}
