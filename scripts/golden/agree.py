"""Expected outputs of indra_ocr.agree for the TS port (tests/indra/layer-check.test.ts).

Run by hand from the repo root:

    uvx --from indra-ocr==1.2.0 python -I scripts/golden/agree.py

Writes tests/fixtures/golden/agree.json: forced_gaps on random recognizer
outputs (one batch, as layer_gaps calls it), and snippet, layer_text and
target_ids on crafted layer tokens with the v1.2 vocabularies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from indra_ocr import agree as A
from indra_ocr.vocab import Vocab

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "fixtures" / "golden" / "agree.json"


def main() -> None:
    rng = np.random.default_rng(11)
    items = []
    for _ in range(12):
        T = int(rng.integers(1, 40))
        K = 8
        logits = rng.normal(size=(T, 30)).astype(np.float32) * 3
        lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        blank = lp[:, 0].astype(np.float32)
        topi = np.argsort(-lp, -1)[:, :K].astype(np.int64)
        topv = np.take_along_axis(lp, topi, -1).astype(np.float32)
        L = int(rng.integers(0, 12))
        ids = []
        for _ in range(L):
            r = rng.random()
            if r < 0.1:
                ids.append(())
            elif r < 0.3:
                ids.append(tuple(int(x) for x in rng.choice(np.arange(1, 30), 2, replace=False)))
            else:
                ids.append((int(rng.integers(1, 30)),))
        items.append((blank, topv, topi, ids))
    gaps = A.forced_gaps(items)
    meta = json.loads((ROOT / "models" / "v1.2" / "meta.json").read_text(encoding="utf-8"))
    vocabs = {g: Vocab.from_json(meta[k]) for g, k in (("base", "vocab"), ("rtl", "vocab_rtl"), ("cjk", "vocab_cjk"))}
    texts = [
        ("base", "Hello world", False),
        ("base", "\u00c9cole \u00e0 Paris", False),
        ("base", "\u1ebfn \u1ed3 Vi\u1ec7t", False),
        ("base", "\u0395\u03bb\u03bb\u03ac\u03b4\u03b1 x \u2212 2", False),
        ("rtl", "\u0645\u0631\u062d\u0628\u0627 12", True),
        ("rtl", "\ufef7\ufe8e", True),
        ("cjk", "\ud55c\uad6d\uc5b4 \u6f22\u5b57", False),
        ("base", "dk;kZy; izkpk;Z", False),
        ("base", "\u0915\u093e\u0930\u094d\u092f", False),
    ]
    targets = [
        {"vocab": g, "text": t, "visual": v, "ids": [list(a) for a in A.target_ids(t, vocabs[g], v)], "writes": A.writes(t, vocabs[g])}
        for g, t, v in texts
    ]
    # Layer tokens (char, x, content index): two runs, spaces, one out of x order.
    toks = []
    x = 0.0
    for run, word in enumerate(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]):
        for c in word:
            toks.append((c, x, run % 3))
            x += 7.5
        toks.append((" ", x, run % 3))
        x += 7.5
    toks.append(("\u0301", 12.0, 0))
    snippets = []
    for n in (3, 8, 12, 24, 60):
        cut_toks, cut = A.snippet(toks, n)
        snippets.append({"n": n, "len": len(cut_toks), "cut": cut})
    layer = [{"rtl": r, "text": A.layer_text(toks, r)} for r in (False, True)]
    OUT.write_text(
        json.dumps(
            {
                "gap_items": [
                    {"blank": b.tolist(), "topv": tv.tolist(), "topi": ti.tolist(), "ids": [list(a) for a in i]}
                    for b, tv, ti, i in items
                ],
                "gaps": [g if np.isfinite(g) else "inf" for g in gaps],
                "targets": targets,
                "tokens": [list(t) for t in toks],
                "snippets": snippets,
                "layer_text": layer,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"{len(items)} gap items, {len(targets)} targets")


if __name__ == "__main__":
    main()
