import { describe, expect, test } from "bun:test";

import { latinShare, layerAgrees } from "../../src/indra/layer-check.js";

// Devanagari from code points: source files stay ASCII.
const deva = (...cps: number[]): string => String.fromCodePoint(...cps);
const KARYALAYA = deva(0x915, 0x93e, 0x930, 0x94d, 0x92f, 0x93e, 0x932, 0x92f); // "office"
const FILE = deva(0x92b, 0x93c, 0x93e, 0x907, 0x932); // "file"

describe("text-layer script check", () => {
  test("latinShare counts basic Latin letters among all letters", () => {
    // Kruti Dev codes for a Hindi line: all Latin.
    expect(latinShare("dk;kZy; izkpk;Z] jktdh; egkfo|ky;")).toBe(1);
    expect(latinShare(KARYALAYA)).toBe(0);
    const mixed = latinShare(`PDF ${FILE}`);
    expect(mixed).toBeGreaterThan(0.3);
    expect(mixed).toBeLessThan(0.6);
    expect(latinShare("12 - 34")).toBe(0);
  });

  test("the page fails only when almost no line agrees", () => {
    expect(layerAgrees([false, false, false, false, false, false])).toBe(false); // a legacy font
    expect(layerAgrees([false, false, false, false, true])).toBe(false); // one Latin title of five
    expect(layerAgrees([true, false, false, false])).toBe(true); // a quarter agree
    expect(layerAgrees([true, true, true, true, true, true])).toBe(true);
    expect(layerAgrees([])).toBe(true); // nothing checked: keep the layer
  });
});
