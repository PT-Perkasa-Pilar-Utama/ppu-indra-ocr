// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/pipeline.py (Apache-2.0, Copyright Datalab).

import type { DetBox } from "./det-post.js";
import type { Gray } from "./image.js";
import type { Box } from "./rec-preproc.js";
import type { RecLine, Straightened } from "./types.js";
import { VERTICAL, rot90 } from "./upright.js";
import { item } from "./util.js";

/**
 * When orient.onnx is unsure but puts at least this much probability on
 * "text runs top to bottom" (a quarter turn either way, or vertical columns),
 * a few lines are read each way and the reading the recognizers are surest
 * of picks the turn. At a thumbnail, vertical CJK columns and a page turned a
 * quarter turn look alike: only the characters tell them apart.
 */
export const SIDEWAYS_MASS = 0.8;
/** Lines read per candidate turn. */
const SIDEWAYS_LINES = 8;

/** The pipeline stages the turn checks read lines with. */
export type LineReader = {
  detect(gray: Gray): Promise<DetBox[]>;
  recognize(gray: Gray, boxes: readonly Box[]): Promise<RecLine[]>;
  straighten(gray: Gray, turns: number, skew: number, small?: Gray): Straightened;
};

/** The SIDEWAYS_LINES widest boxes, widest first (a stable sort: numpy's argsort breaks ties its own way). */
function longest(boxes: readonly DetBox[]): Box[] {
  return boxes
    .map((b): Box => [b[0], b[1], b[2], b[3]])
    .toSorted((a, b) => Math.fround(b[2] - b[0]) - Math.fround(a[2] - a[0]))
    .slice(0, SIDEWAYS_LINES);
}

/** The recognizers' mean confidence on these lines (0 without lines). */
async function meanConf(reader: LineReader, gray: Gray, boxes: readonly Box[]): Promise<number> {
  if (!boxes.length) return 0;
  const lines = await reader.recognize(gray, boxes);
  return lines.reduce((s, l) => s + l.conf, 0) / lines.length;
}

/**
 * The turn of a page orient.onnx is unsure of but sees as likely sideways
 * (probabilities `probs`): up to SIDEWAYS_LINES of the longest lines are read
 * as the page stands, turned counter-clockwise, and turned clockwise (the
 * counter-clockwise lines turned half round); the reading the recognizers
 * are surest of picks the side. On the counter-clockwise side, a quarter turn
 * (3) and vertical columns (VERTICAL) give the same crops, so orient.onnx,
 * which sees the characters' shape, picks between them.
 */
export async function readTurn(
  reader: LineReader,
  gray: Gray,
  skew: number,
  small: Gray,
  probs: readonly number[]
): Promise<number> {
  const up = reader.straighten(gray, 0, skew, small).image;
  const ccw = reader.straighten(gray, 3, skew, small).image;
  const sel = longest(await reader.detect(ccw));
  const { width: w, height: h } = ccw;
  const cw = rot90(ccw, 2); // the page turned clockwise
  const f = Math.fround;
  const selCw = sel.map(([x0, y0, x1, y1]): Box => [f(w - x1), f(h - y1), f(w - x0), f(h - y0)]);
  const side = probs.length > VERTICAL && item(probs, VERTICAL) > item(probs, 3) ? VERTICAL : 3;
  // Insertion order breaks ties, as Python's max over a dict: as it stands, clockwise, the side.
  const scores: [number, number][] = [
    [0, await meanConf(reader, up, longest(await reader.detect(up)))],
    [1, await meanConf(reader, cw, selCw)],
    [side, await meanConf(reader, ccw, sel)],
  ];
  let best = item(scores, 0);
  for (const s of scores) if (s[1] > best[1]) best = s;
  return best[1] > 0 ? best[0] : 0;
}

/** 2 when the longest lines read surer with the page turned half round than as it stands, else 0. */
export async function readHalfTurn(
  reader: LineReader,
  gray: Gray,
  skew: number,
  small: Gray
): Promise<number> {
  const conf = async (img: Gray): Promise<number> =>
    meanConf(reader, img, longest(await reader.detect(img)));
  const up = reader.straighten(gray, 0, skew, small).image;
  const down = reader.straighten(gray, 2, skew, small).image;
  return (await conf(down)) > (await conf(up)) ? 2 : 0;
}
