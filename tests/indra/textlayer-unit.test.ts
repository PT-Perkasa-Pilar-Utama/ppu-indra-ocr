import { describe, expect, test } from "bun:test";

import { assignRuns, rightToLeft } from "../../src/indra/document.js";
import type { Box } from "../../src/indra/rec-preproc.js";
import type {
  PdfChar,
  PdfPageSource,
  TextLayerStats,
  TextLine,
} from "../../src/indra/textlayer.js";
import {
  BROKEN_FONT_MAP,
  badCharCount,
  codePoint,
  collapseSpaces,
  pageTurn,
  readTextLayer,
  verticalLayout,
  visualRtlFrac,
} from "../../src/indra/textlayer.js";

const HEB = String.fromCharCode(0x05e9, 0x05dc, 0x05d5, 0x05dd); // a Hebrew word
const run = (text: string, x0: number, seq = 0): TextLine => {
  const chars = [...text].map((c, i): [string, Box] => [c, [x0 + i * 10, 0, x0 + i * 10 + 8, 10]]);
  return { bbox: [x0, 0, x0 + text.length * 10, 10], text, chars, quarter: 0, seq };
};

const stats = (over: Partial<TextLayerStats>): TextLayerStats => ({
  chars: 100,
  rotated: 0,
  bad: 0,
  quarters: [100, 0, 0, 0],
  ...over,
});

describe("right-to-left lines", () => {
  test("rightToLeft counts strong characters", () => {
    expect(rightToLeft([...HEB].map((c) => [c, 0] as const))).toBe(true);
    expect(rightToLeft([..."abc"].map((c) => [c, 0] as const))).toBe(false);
    expect(rightToLeft([[7, 0]])).toBe(false);
  });

  test("assignRuns orders the runs of an RTL line right to left", () => {
    // Two Hebrew runs on one line: the right one reads first.
    const { toks, lost } = assignRuns([run(HEB, 0, 0), run(HEB, 100, 1)], [[0, 0, 200, 10]], 1);
    expect(lost).toBe(0);
    const runOrder = (toks[0] ?? []).flatMap((t) => (t[0] === " " ? [] : [t[2]]));
    expect(runOrder[0]).toBe(1);
  });

  test("characters outside every box are lost; no boxes loses all ink", () => {
    expect(assignRuns([run("ab", 0)], [[500, 500, 600, 600]], 1).lost).toBe(2);
    expect(assignRuns([run("a b", 0)], [], 1)).toEqual({ toks: [], lost: 2 });
  });
});

describe("text-layer character checks", () => {
  test("badCharCount flags replacement, private-use, lone surrogates and controls", () => {
    expect(badCharCount("ok\t")).toBe(0);
    expect(badCharCount(String.fromCharCode(0xfffd))).toBe(1);
    expect(badCharCount(String.fromCharCode(0xe000))).toBe(1);
    expect(badCharCount(String.fromCharCode(0xd800))).toBe(1);
    expect(badCharCount(String.fromCharCode(0x01))).toBe(1);
  });

  test("badCharCount takes a character above U+FFFF as one good character", () => {
    expect(badCharCount(String.fromCodePoint(0x1d44e))).toBe(0);
  });

  test("codePoint joins a surrogate pair and replaces a lone surrogate", () => {
    expect(codePoint(0xd835, 0xdc4e)).toEqual([String.fromCodePoint(0x1d44e), true]);
    expect(codePoint(0xd835, 0x41)).toEqual([String.fromCharCode(0xfffd), false]);
    expect(codePoint(0x41)).toEqual(["A", false]);
  });

  test("collapseSpaces joins words with single spaces, as Python's split()", () => {
    expect(collapseSpaces("  a \t b\n")).toBe("a b");
    expect(collapseSpaces("")).toBe("");
  });
});

describe("turned and visual-order layers", () => {
  test("pageTurn picks the quarter turn that holds most characters", () => {
    expect(pageTurn(stats({ quarters: [9, 159, 0, 0], chars: 168 }))).toBe(1);
    expect(pageTurn(stats({ quarters: [100, 10, 0, 0] }))).toBe(0);
    expect(pageTurn(stats({ quarters: [0, 0, 0, 376], chars: 376 }))).toBe(3);
  });

  test("verticalLayout needs most steps to go down", () => {
    expect(verticalLayout(stats({ steps: [92, 0] }))).toBe(true);
    expect(verticalLayout(stats({ steps: [29, 0] }))).toBe(false);
    expect(verticalLayout(stats({ steps: [40, 30] }))).toBe(false);
  });

  test("visualRtlFrac counts right-to-left letters stored left to right", () => {
    // Characters placed left to right: every Hebrew pair steps right.
    const visual = run(HEB.repeat(8), 0);
    expect(visualRtlFrac([visual])).toEqual({ pairs: 31, share: 1 });
    // Logical order: each next letter sits to the left.
    const logical: TextLine = {
      ...visual,
      chars: visual.chars.map(([c, [x0, y0, x1, y1]]): [string, Box] => [
        c,
        [1000 - x1, y0, 1000 - x0, y1],
      ]),
    };
    expect(visualRtlFrac([logical]).share).toBe(0);
  });
});

describe("readTextLayer sets bad-code runs apart", () => {
  // A 200 x 100 pt page, upright, every pixel dark (each run has ink).
  const fakePage = (lines: string[]): PdfPageSource => {
    const chars: PdfChar[] = [];
    for (const [row, text] of lines.entries()) {
      for (const [i, c] of [...text].entries()) {
        const l = 10 + 8 * i;
        const b = 80 - 20 * row;
        chars.push({ unit: c.charCodeAt(0), angle: 0, box: [l, b, l + 6, b + 10] });
      }
      chars.push({ unit: 10, angle: 0, box: undefined });
    }
    return {
      size: () => [200, 100],
      render: (scale) => {
        const width = Math.ceil(200 * scale);
        const height = Math.ceil(100 * scale);
        return { data: new Uint8Array(width * height), width, height };
      },
      chars: () => chars,
      pageToDevice: (w, h, x, y) => [Math.round((x * w) / 200), Math.round(((100 - y) * h) / 100)],
      textRenderModes: () => [0],
      rotation: () => 0,
      setRotation: () => undefined,
      symbolDigitBoxes: () => [],
    };
  };
  const pua = String.fromCharCode(0xe001, 0xe002, 0xe003);
  const dark = { data: new Uint8Array(200 * 100), width: 200, height: 100 };

  test("an isolated run of private-use codes goes to OCR, the rest of the layer stays", () => {
    const tl = readTextLayer(fakePage(["Hello world here", pua]), dark, 1);
    expect(tl.trusted).toBe(true);
    expect(tl.rejected.map((r) => r.text)).toEqual([pua]);
    expect(tl.lines.map((r) => r.text)).toEqual(["Hello world here"]);
    expect(tl.stats.bad_runs).toBe(1);
  });

  test("bad codes inside word text mark the whole layer as a broken font map", () => {
    const tl = readTextLayer(fakePage(["Hello world here", `ab${pua}`]), dark, 1);
    expect(tl.trusted).toBe(false);
    expect(tl.reason).toBe(BROKEN_FONT_MAP);
  });
});
