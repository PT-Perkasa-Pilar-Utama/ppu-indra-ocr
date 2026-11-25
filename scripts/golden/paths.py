"""Expected outputs for 1.2 code paths the page goldens do not reach, for the TS port.

Run by hand from the repo root:

    INDRA_OCR_NATIVE=0 uvx --from indra-ocr==1.2.0 python -I scripts/golden/paths.py

Writes tests/fixtures/golden/paths.json:
- read_turn on golden pages with orientation probabilities that send a page
  to the check (orient.onnx unsure, mostly sideways);
- MathReader.accept on the recorded readings as table lines (in_table 1);
- math_span_img.splice / unspliced and OrtOCR._decode on crafted tokens;
- document.symbol_words and symbol_candidates on crafted lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import indra_ocr
from indra_ocr import document as D
from indra_ocr import math_span_img as S
from indra_ocr import pipeline as P

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "fixtures" / "golden"


def tok_json(toks):
    out = []
    for t in toks:
        kind = "formula" if isinstance(t[0], S.Formula) else "pad" if isinstance(t[0], S.Pad) else None
        out.append([str(t[0]), round(float(t[1]), 4), kind])
    return out


def main() -> None:
    conv = indra_ocr.load(path=str(ROOT / "models" / "v1.2"), threads=1)
    ocr = conv.ocr
    turns = []
    probs = {
        "sideways": [0.05, 0.45, 0.0, 0.45, 0.05],
        "vertical": [0.05, 0.1, 0.0, 0.1, 0.75],
    }
    for name in ("rotated-90", "two-column", "vertical", "chinese"):
        gray = cv2.imread(str(GOLDEN / f"{name}.p0.gray.png"), cv2.IMREAD_GRAYSCALE)
        small = P.preview(gray)
        a = P.estimate_skew(small)
        for label, p in probs.items():
            k = ocr.read_turn(gray, a, small, np.array(p, np.float32))
            turns.append({"page": f"{name}.p0.gray.png", "probs": label, "p": p, "skew": a, "k": int(k)})
    events = []
    for f in sorted(GOLDEN.glob("*.json")):
        data = json.loads(f.read_text())
        if isinstance(data, dict):
            events += [e for e in data.get("events", []) if e["stage"] == "math.accept"]
    accept = [
        {**{k: e[k] for k in ("text", "conf", "draft", "rconf")}, "out": ocr.math.accept(e["text"], e["conf"], e["draft"], e["rconf"], in_table=1.0)}
        for e in events
    ]
    cyr = list("\u0444\u043e\u0440\u043c\u0443\u043b\u0430 x2 + y \u0432\u0435\u0440\u043d\u0430")
    base = [(c, 10.0 + 6 * i) for i, c in enumerate(cyr)]
    arabic = list("\u0627\u0644\u0645\u0639\u0627\u062f\u0644\u0629 x2 \u0635\u062d\u064a\u062d\u0629")
    rtl = [(c, 10.0 + 6 * i) for i, c in enumerate(arabic)]
    splices = []
    for label, toks, x0, x1, latex, page_rtl in (
        ("cyrillic", base, 52.0, 88.0, "$x ^ { 2 } + y$", False),
        ("cyrillic-edge", base, 0.0, 30.0, "$a$", False),
        ("arabic", rtl, 58.0, 70.0, "$x ^ { 2 }$", True),
    ):
        sp = S.splice(toks, x0, x1, latex)
        ocr._page_rtl = page_rtl
        splices.append(
            {
                "label": label,
                "toks": [[c, x] for c, x in toks],
                "x0": x0,
                "x1": x1,
                "latex": latex,
                "page_rtl": page_rtl,
                "spliced": tok_json(sp),
                "unspliced": tok_json(S.unspliced(sp)),
                "decoded": ocr._decode(sp),
                "decoded_plain": ocr._decode(S.unspliced(sp)),
            }
        )
    ocr._page_rtl = False
    sym = np.array([[40.0, 10.0, 47.0, 22.0], [100.0, 11.0, 108.0, 21.0]], np.float32)
    line = [(c, 10.0 + 6 * i, 0) for i, c in enumerate("x 2 5 = 7 and 12")]
    words = D.symbol_words(line, [8.0, 9.0, 120.0, 23.0], sym)
    cands = [
        {"layer": layer, "read": read, "hits": hits, "out": {str(k): v for k, v in D.symbol_candidates(layer, read, hits).items()}}
        for layer, read, hits in (
            ("x 2 5 = 7", "x \u2212 5 = 7", [2]),
            ("a 6 b", "a \u00b1 b", [2]),
            ("a 6 b", "a . b", [2]),
            ("10 2 3", "10 \u2013 3", [3]),
            ("x 2 5", "x 2 5", [2]),
        )
    ]
    (GOLDEN / "paths.json").write_text(
        json.dumps(
            {
                "read_turn": turns,
                "accept_in_table": accept,
                "splices": splices,
                "symbol_words": {"line": [list(t) for t in line], "box": [8.0, 9.0, 120.0, 23.0], "sym": sym.tolist(), "words": [[a, b, h] for a, b, h in words]},
                "symbol_candidates": cands,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    print(len(turns), "turns,", len(accept), "accepts,", len(splices), "splices")


if __name__ == "__main__":
    main()
