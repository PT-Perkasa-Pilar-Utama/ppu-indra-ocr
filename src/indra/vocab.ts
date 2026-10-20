// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/vocab.py (Apache-2.0, Copyright Datalab).

const LETTERS = /^\p{L}+$/u;

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
