import { describe, expect, test } from "bun:test";

import { joinLines, pageMarkdown, tableHtml } from "../../src/indra/layout-post.js";
import type { Box } from "../../src/indra/rec-preproc.js";

/**
 * Hand-built inputs for markdown branches the page fixtures do not reach:
 * lists, display math, page furniture, table titles and wrapped cells.
 */

const CLASSES = 10;

/** Class logits [n, 10] with one class per line, and same-block logits for the given blocks. */
type Layout = { cls: Float32Array; same: Float32Array };

function layout(labels: number[], blocks: number[][]): Layout {
  const n = labels.length;
  const cls = new Float32Array(n * CLASSES).fill(-5);
  for (const [i, c] of labels.entries()) cls[i * CLASSES + c] = 5;
  const same = new Float32Array(n * n).fill(-5);
  for (const b of blocks) for (const i of b) for (const j of b) same[i * n + j] = 5;
  return { cls, same };
}

const rows = (n: number): Box[] =>
  Array.from({ length: n }, (_, i): Box => [0, i * 20, 200, i * 20 + 10]);

describe("joinLines", () => {
  test("joins a hyphenated word across lines, and CJK lines without a space", () => {
    expect(joinLines(["tumori-", "genesis is", "rare"])).toBe("tumorigenesis is rare");
    const han = String.fromCharCode(0x6c34);
    expect(joinLines([han, han])).toBe(han + han);
  });
});

describe("pageMarkdown", () => {
  test("a List-Group block writes one item per marker; a continuation joins its item", () => {
    const texts = ["- one", "continued", "- two"];
    const { cls, same } = layout([4, 4, 4], [[0, 1, 2]]);
    expect(pageMarkdown(rows(3), texts, [0, 1, 2], cls, same)).toBe("- one continued\n- two");
  });

  test("a block of whole formulas becomes display math", () => {
    const { cls, same } = layout([0, 0], [[0, 1]]);
    expect(pageMarkdown(rows(2), ["$x^2$", "$y$"], [0, 1], cls, same)).toBe("$$x^2$$\n$$y$$");
  });

  test("page headers and footers are left out unless kept", () => {
    const { cls, same } = layout([2, 0, 3], [[0], [1], [2]]);
    const texts = ["Header", "Body", "7"];
    expect(pageMarkdown(rows(3), texts, [0, 1, 2], cls, same)).toBe("Body");
    expect(pageMarkdown(rows(3), texts, [0, 1, 2], cls, same, true)).toBe("Header\n\nBody\n\n7");
  });
});

describe("tableHtml", () => {
  test("a first row spanning the table is a title before it", () => {
    const boxes: Box[] = [
      [0, 0, 300, 10],
      [0, 20, 100, 30],
      [200, 20, 300, 30],
      [0, 40, 100, 50],
      [200, 40, 300, 50],
    ];
    const html = tableHtml(boxes, ["Table 1", "Name", "Count", "a", "1"]);
    expect(html.startsWith("Table 1\n\n<table>")).toBe(true);
    expect(html).toContain("<tr><th>Name</th><th>Count</th></tr>");
    expect(html).toContain("<tr><td>a</td><td>1</td></tr>");
  });

  test("a wrapped cell's second line joins its row; text is escaped", () => {
    const boxes: Box[] = [
      [0, 0, 100, 10],
      [200, 0, 300, 10],
      [0, 30, 100, 40],
      [200, 30, 300, 40],
      [0, 42, 100, 52], // continuation: a small gap, fewer cells than the row above
      [0, 72, 100, 82],
      [200, 72, 300, 82],
    ];
    const html = tableHtml(boxes, ["Item", "Price", "long", "5", "name", "b<c", "6"]);
    expect(html).toContain("<tr><td>long name</td><td>5</td></tr>");
    expect(html).toContain("<td>b&lt;c</td>");
  });
});
