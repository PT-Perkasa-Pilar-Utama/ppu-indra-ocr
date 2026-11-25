// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama
// Ported from datalab-to/indra indra_ocr/document.py (Apache-2.0, Copyright Datalab).

import type { Gray } from "./image.js";
import { latinLine } from "./math-text.js";
import type { IndraPipeline } from "./pipeline.js";
import type { Box } from "./rec-preproc.js";
import { cropLine } from "./rec-preproc.js";
import type { RecLine } from "./types.js";
import { item } from "./util.js";

/**
 * Text-layer lines that hold math (the classifier, on their crops) are read
 * by the math reader: the layer has the symbols but not the LaTeX. Only lines
 * whose letters are mostly Latin go (MathRec writes Latin letters); a learned
 * gate may skip some, weighing text-layer and hybrid pages apart. An accepted
 * reading takes the layer's accents in its prose, and the line keeps its
 * place in the content order (its first character's index).
 */
export async function readLayerMath(
  pipeline: IndraPipeline,
  gray: Gray,
  boxes: readonly Box[],
  rec: (RecLine | undefined)[],
  source: "text" | "hybrid"
): Promise<void> {
  const math = pipeline.mathReader;
  if (!math) return;
  const idx = rec.flatMap((r, i) => {
    const first = r?.toks[0];
    return r && first && typeof first[0] === "string" && first[2] !== undefined && latinLine(r.text)
      ? [i]
      : [];
  });
  if (!idx.length) return;
  const crops = idx.map((i) => cropLine(gray, item(boxes, i), pipeline.height, pipeline.imageOps));
  const { flags, probs } = await math.flags(crops);
  let flagged = idx.filter((_, n) => flags[n]);
  const draftOf = (i: number): string => item(rec, i)?.text ?? "";
  if (flagged.length && math.hasGate) {
    const keep = math.gate(
      probs.filter((_, n) => flags[n]),
      flagged.map(draftOf),
      idx.length,
      source
    );
    flagged = flagged.filter((_, n) => keep[n]);
  }
  if (!flagged.length) return;
  const readings = await math.read(
    gray,
    flagged.map((i) => item(boxes, i))
  );
  for (const [n, i] of flagged.entries()) {
    const { text, conf } = item(readings, n);
    const r = rec[i];
    if (!r) continue;
    const merged = math.accept(text, conf, r.text, 1.0);
    if (merged === null) continue;
    const first = Math.min(...r.toks.flatMap((tk) => (tk[2] === undefined ? [] : [tk[2]])));
    const [x0, , x1] = item(boxes, i);
    rec[i] = {
      text: merged,
      conf: 1,
      toks: [[merged, Math.fround(x0 + x1) / 2, first]],
      math: { plain: r, reading: text, conf, draft: r.text, rconf: 1.0 },
    };
  }
}
