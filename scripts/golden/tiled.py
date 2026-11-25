"""A golden for tiled detection (pipeline._detect_tiled) for the TS port.

Run by hand from the repo root:

    INDRA_OCR_NATIVE=0 uvx --from indra-ocr==1.2.0 python -I scripts/golden/tiled.py

Draws a 4400 x 3000 page with small text (a plan sheet's labels), checks that
the reference detects it in tiles, and writes tests/fixtures/golden/tiled.gray.png
(the page) and tests/fixtures/golden/tiled.json (its boxes from OrtOCR.detect).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from indra_ocr import pipeline as P

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "fixtures" / "golden"


def page() -> np.ndarray:
    rng = np.random.default_rng(5)
    gray = np.full((3000, 4400), 255, np.uint8)
    words = ["ROOM", "DOOR", "STAIR", "WC", "KITCHEN", "HALL", "12.40", "3.05", "N", "OFFICE"]
    for _ in range(260):
        x, y = int(rng.integers(20, 4200)), int(rng.integers(20, 2980))
        text = " ".join(rng.choice(words, size=int(rng.integers(1, 4))))
        cv2.putText(gray, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    # a few rules, as a drawing has
    for k in range(12):
        cv2.line(gray, (0, 240 * k + 5), (4400, 240 * k + 5), 0, 2)
    return gray


def main() -> None:
    gray = page()
    ocr = P.OrtOCR(ROOT / "models" / "v1.2", threads=1)
    tiled = []
    inner = ocr._detect_tiled

    def wrap(g):
        b = inner(g)
        tiled.append(len(b))
        return b

    ocr._detect_tiled = wrap
    boxes = ocr.detect(gray)
    assert tiled, "the page did not take the tiled path"
    cv2.imwrite(str(OUT / "tiled.gray.png"), gray)
    (OUT / "tiled.json").write_text(
        json.dumps({"tiled": True, "boxes": [[round(float(v), 3) for v in b] for b in boxes]}) + "\n"
    )
    print(f"{len(boxes)} boxes, tiled")


if __name__ == "__main__":
    main()
