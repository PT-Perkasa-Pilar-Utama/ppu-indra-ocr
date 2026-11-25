// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/vocab.py (Apache-2.0, Copyright Datalab).

import { isPySpace } from "./util.js";

const LETTERS = /^\p{L}+$/u;

/**
 * Text canonicalization: characters the image cannot tell apart map to one
 * form. Romanian comma-below and Turkish cedilla look the same in most fonts;
 * the font decides a bullet's drawn shape.
 */
const CANON = new Map(
  [
    [0x218, 0x15e],
    [0x219, 0x15f],
    [0x21a, 0x162],
    [0x21b, 0x163],
    [0x25aa, 0x2022],
    [0x25cf, 0x2022],
    [0x2219, 0x2022],
    [0x2023, 0x2022],
    [0x2043, 0x2022],
  ].map(([a = 0, b = 0]) => [String.fromCharCode(a), String.fromCharCode(b)])
);
/** Runs of 4+ leader or rule characters: the count cannot be read from the image. */
const RUNS = new RegExp("([._\\-=\\u00b7*~])\\1{3,}", "gu");
const DOUBLE_SINGLE = /''|``/gu;
/** A token: a LaTeX command, an escaped symbol, or one character. */
const TOKEN = /\\[A-Za-z]+|\\[{}|,;! ]|./gsu;
/** $...$ is math only when it holds LaTeX, so prices ("$5 or $10") stay text. */
const MATH_SPAN = /\$[^$]*[\\^_{][^$]*\$/gu;

/** `vocab.canonical`: NFC, the CANON map, and leader runs cut to three. */
export function canonical(text: string): string {
  const mapped = Array.from(text.normalize("NFC"), (c) => CANON.get(c) ?? c).join("");
  return mapped.replace(DOUBLE_SINGLE, '"').replace(RUNS, (_, c: string) => c.repeat(3));
}

/**
 * Token vocabulary of a line recognizer. Class 0 is the CTC blank; class i + 1
 * is `chars[i]`. A token is one character, or one LaTeX command inside math.
 */
export class Vocab {
  public readonly chars: readonly string[];
  public readonly index: ReadonlyMap<string, number>;
  public readonly hasCommands: boolean;
  /** Hangul syllables are read as jamo and composed back (NFC) on decode. */
  public readonly jamo: boolean;

  public constructor(chars: readonly string[]) {
    this.chars = chars;
    this.index = new Map(chars.map((c, i) => [c, i + 1]));
    this.hasCommands = chars.some((c) => [...c].length > 1);
    this.jamo = this.index.has(String.fromCharCode(0x1100));
  }

  /** A vocabulary from a `vocab` / `vocab_<group>` JSON string in meta.json. */
  public static fromJson(json: string): Vocab {
    // SAFETY: meta.json ships with the models; the shape is checked below.
    const data = JSON.parse(json) as { blank?: number; chars?: unknown };
    if ((data.blank ?? 0) !== 0) throw new Error("only blank=0 vocabularies are supported");
    if (!Array.isArray(data.chars) || !data.chars.every((c) => typeof c === "string")) {
      throw new Error("vocab JSON has no chars array");
    }
    return new Vocab(data.chars);
  }

  /** The text as the recognizer's targets write it (`canonical`). */
  public normalize(text: string): string {
    return canonical(text);
  }

  /**
   * Characters, with a LaTeX command as one token when the vocab has it.
   * Inside a $...$ math span spaces are dropped: they are not visible in
   * rendered math.
   */
  public tokenize(text: string): string[] {
    if (!this.hasCommands) return Array.from(text);
    const out: string[] = [];
    let pos = 0;
    for (const m of text.matchAll(MATH_SPAN)) {
      out.push(...this.tokens(text.slice(pos, m.index), false), ...this.tokens(m[0], true));
      pos = m.index + m[0].length;
    }
    out.push(...this.tokens(text.slice(pos), false));
    return out;
  }

  private tokens(text: string, math: boolean): string[] {
    const out: string[] = [];
    for (const [t] of text.matchAll(TOKEN)) {
      if (math && isPySpace(t)) continue;
      if (this.index.has(t) || Array.from(t).length === 1) out.push(t);
      else out.push(...Array.from(t));
    }
    return out;
  }

  /** Text of class ids. A space goes back between a command and a following letter. */
  public decode(ids: readonly number[]): string {
    const toks: string[] = [];
    for (const id of ids) {
      const tok = this.chars[id - 1];
      if (id > 0 && tok !== undefined) toks.push(tok);
    }
    if (this.jamo) return toks.join("").normalize("NFC");
    if (!this.hasCommands) return toks.join("");
    let out = "";
    for (const [k, t] of toks.entries()) {
      out += t;
      // "\alpha" + "x" must not read back as "\alphax"
      const next = toks[k + 1] ?? "";
      if (
        t.length > 2 &&
        t[0] === "\\" &&
        LETTERS.test(t.slice(1)) &&
        LETTERS.test([...next][0] ?? "")
      ) {
        out += " ";
      }
    }
    return out;
  }
}
