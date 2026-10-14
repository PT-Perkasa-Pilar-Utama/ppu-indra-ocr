import { describe, expect, test } from "bun:test";

import { assignRuns, rightToLeft } from "../../src/indra/document.js";
import type { Box } from "../../src/indra/rec-preproc.js";
import { badCharCount, collapseSpaces } from "../../src/indra/textlayer.js";
import type { TextLine } from "../../src/indra/textlayer.js";

const HEB = String.fromCharCode(0x05e9, 0x05dc, 0x05d5, 0x05dd); // a Hebrew word
const run = (text: string, x0: number): TextLine => {
  const chars = [...text].map((c, i): [string, Box] => [c, [x0 + i * 10, 0, x0 + i * 10 + 8, 10]]);
  return { bbox: [x0, 0, x0 + text.length * 10, 10], text, chars };
};

describe("right-to-left lines", () => {
  test("rightToLeft counts strong characters", () => {
    expect(rightToLeft([...HEB].map((c) => [c, 0] as const))).toBe(true);
    expect(rightToLeft([..."abc"].map((c) => [c, 0] as const))).toBe(false);
    expect(rightToLeft([[7, 0]])).toBe(false);
  });

  test("assignRuns orders the runs of an RTL line right to left", () => {
    // Two Hebrew runs on one line: the right one reads first.
    const { toks, lost } = assignRuns([run(HEB, 0), run(HEB, 100)], [[0, 0, 200, 10]], 1);
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

  test("collapseSpaces joins words with single spaces, as Python's split()", () => {
    expect(collapseSpaces("  a \t b\n")).toBe("a b");
    expect(collapseSpaces("")).toBe("");
  });
});
