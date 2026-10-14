"""Dump stage-by-stage goldens from the Python reference (indra-ocr 0.3.1).

Run by hand from the repo root:

    uvx --from indra-ocr==0.3.1 python -I scripts/golden/gen.py

It reads the models in models/ (the v3.1 mirror) and the fixtures in
tests/fixtures/pages/ and assets/, and writes tests/fixtures/golden/:

- <case>.json: every stage the reference ran on each page, in call order,
  then the final Page fields.
- <input stem>.p<N>.gray.png: the grayscale page the OCR read (after decode or PDF
  render, before orient and deskew). TS stage tests start from these bytes,
  so JPEG decoders and PDF renderers cannot hide a pipeline difference.

CI never runs this script; the output is committed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import indra_ocr
from indra_ocr import document as D
from indra_ocr import pipeline as P
from indra_ocr.textlayer import order_backtracks

ROOT = Path(__file__).resolve().parents[2]
HAL = next(
    (ln.split(":", 1)[1].strip() for ln in cv2.getBuildInformation().splitlines() if "Custom HAL" in ln),
    "none",
)
MODELS = ROOT / "models"
PAGES = ROOT / "tests" / "fixtures" / "pages"
OUT = ROOT / "tests" / "fixtures" / "golden"

# case name -> (input file, load() options)
CASES = {
    "receipt": (ROOT / "assets" / "receipt.jpg", {}),
    "tilted": (ROOT / "assets" / "tilted.png", {}),
    "two-column": (PAGES / "two-column.png", {}),
    "rotated-90": (PAGES / "rotated-90.png", {}),
    "rotated-180": (PAGES / "rotated-180.png", {}),
    "table": (PAGES / "table.png", {}),
    "chinese": (PAGES / "chinese.png", {}),
    "math": (PAGES / "math.png", {}),
    "math.on": (PAGES / "math.png", {"math": True}),
    "digital.auto": (PAGES / "digital.pdf", {}),
    "digital.ocr": (PAGES / "digital.pdf", {"mode": "ocr"}),
    "scanned.auto": (PAGES / "scanned.pdf", {}),
}


def r(x, n: int = 4):
    """JSON-friendly copy of numpy data, floats rounded to n places."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [r(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), n)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Recorder:
    """Wraps the stage methods of one OrtOCR and records each call."""

    def __init__(self, ocr: P.OrtOCR, case: str):
        self.ocr, self.case = ocr, case
        self.events: list[dict] = []
        self._wrap_sessions()
        self._wrap_methods()

    def log(self, stage: str, **data) -> None:
        self.events.append({"stage": stage, **data})

    def _wrap_sessions(self) -> None:
        rec = self

        class Session:
            def __init__(self, name: str, sess):
                self.name, self.sess = name, sess

            def get_inputs(self):
                return self.sess.get_inputs()

            def get_outputs(self):
                return self.sess.get_outputs()

            def run(self, names, feed):
                outs = self.sess.run(names, feed)
                rec.on_run(self.name, feed, outs)
                return outs

        for name in ("det", "rec", "rec_cjk", "script", "order", "orient"):
            sess = getattr(self.ocr, name, None)
            if sess is not None:
                setattr(self.ocr, name, Session(name, sess))

    def on_run(self, name: str, feed: dict, outs: list) -> None:
        if name == "det":
            x = feed["x"]
            self.log("det.run", input_shape=list(x.shape), input_sha=sha(x), output_shape=list(outs[0].shape))
        elif name == "orient":
            self.log("orient.run", input_sha=sha(feed["x"]), p=r(outs[0][0], 6))
        elif name == "script":
            self.log("script.run", input_shape=list(feed["x"].shape), widths=r(feed["widths"]), p=r(outs[0], 6))
        elif name in ("rec", "rec_cjk"):
            self.log(
                f"{name}.run",
                input_shape=list(feed["x"].shape),
                input_sha=sha(feed["x"]),
                widths=r(feed["widths"]),
                lengths=r(outs[2]),
            )
        elif name == "order":
            outputs = {"order.a_shape": list(outs[0].shape)}
            if len(outs) >= 3:
                outputs["cls_argmax"] = r(outs[1][0].argmax(-1))
            self.log("order.run", feats=r(feed["feats"][0], 6), **outputs)

    def _wrap_methods(self) -> None:
        ocr, rec = self.ocr, self

        upright = ocr.upright

        def upright_(gray, turn=True):
            out, k, a = upright(gray, turn)
            rec.log("upright", turn=turn, k=k, skew=r(a, 6), shape=list(out.shape), sha=sha(out))
            return out, k, a

        ocr.upright = upright_

        detect_at = ocr._detect_at

        def detect_at_(gray, long_side):
            boxes, s = detect_at(gray, long_side)
            rec.log("detect_at", long_side=long_side, scale=r(s, 8), boxes=r(boxes, 3))
            return boxes, s

        ocr._detect_at = detect_at_

        detect = ocr.detect

        def detect_(gray):
            boxes = detect(gray)
            rec.log("detect", shape=list(gray.shape), boxes=r(boxes, 3))
            return boxes

        ocr.detect = detect_

        recognize = ocr.recognize

        def recognize_(gray, boxes, batch=64):
            out = recognize(gray, boxes, batch)
            lines = [{"text": t, "conf": r(c, 6), "toks": r(tk, 3)} for t, c, tk in out]
            rec.log("recognize", boxes=r(boxes[:, :4], 3), lines=lines)
            return out

        ocr.recognize = recognize_

        reading_order = ocr.reading_order

        def reading_order_(boxes, texts, wh):
            order, layout, logits = reading_order(boxes, texts, wh)
            data = {"boxes": r(boxes[:, :4], 3), "texts": texts, "wh": list(wh), "order": r(order)}
            if layout is not None:
                data["cls_argmax"] = r(layout[0].argmax(-1))
            rec.log("reading_order", **data)
            return order, layout, logits

        ocr.reading_order = reading_order_

        math = getattr(ocr, "math", None)
        if math is not None:
            flags = math.flags

            def flags_(crops, batch=64):
                out = flags(crops, batch)
                rec.log("math.flags", widths=[int(c.shape[1]) for c in crops], flags=out)
                return out

            math.flags = flags_
            read = math.read

            def read_(gray, boxes, batch=64):
                out = read(gray, boxes, batch)
                rec.log("math.read", boxes=r(boxes[:, :4], 3), lines=[{"text": t, "conf": r(c, 6)} for t, c in out])
                return out

            math.read = read_


class DocumentRecorder:
    """Wraps the PDF text-layer functions of indra_ocr.document for one case."""

    def __init__(self, rec: Recorder):
        self.rec = rec
        self.saved = {name: getattr(D, name) for name in ("_render", "read_text_layer", "assign_runs", "readable")}
        self.saved_coherent = P.pdf_order_coherent

    def __enter__(self):
        rec, saved = self.rec, self.saved

        def render_(page, scale):
            out = saved["_render"](page, scale)
            rec.log("render", scale=r(scale, 8), shape=list(out.shape), sha=sha(out))
            return out

        def read_text_layer_(page, gray, scale):
            tl = saved["read_text_layer"](page, gray, scale)
            rec.log(
                "read_text_layer",
                size=r(tl.size, 4),
                trusted=tl.trusted,
                reason=tl.reason,
                stats=tl.stats,
                runs=[{"bbox": r(ln.bbox, 3), "text": ln.text, "chars": [[c, r(b, 3)] for c, b in ln.chars]} for ln in tl.lines],
            )
            return tl

        def assign_runs_(runs, boxes, k):
            toks, lost = saved["assign_runs"](runs, boxes, k)
            rec.log("assign_runs", k=r(k, 8), boxes=r(boxes[:, :4], 3), lost=lost, toks=[[[c, r(x, 3), i] for c, x, i in tk] for tk in toks])
            return toks, lost

        def readable_(ocr, runs):
            out = saved["readable"](ocr, runs)
            rec.log("readable", value=out)
            return out

        coherent = self.saved_coherent

        def coherent_(boxes, pair_logits=None):
            out = coherent(boxes, pair_logits)
            rec.log("pdf_order_coherent", boxes=r(boxes[:, :4], 3), backtracks=order_backtracks(boxes), value=out)
            return out

        D._render, D.read_text_layer, D.assign_runs, D.readable = render_, read_text_layer_, assign_runs_, readable_
        P.pdf_order_coherent = coherent_
        return self

    def __exit__(self, *exc):
        for name, fn in self.saved.items():
            setattr(D, name, fn)
        P.pdf_order_coherent = self.saved_coherent


def page_json(page) -> dict:
    return {
        "source": page.source,
        "note": page.note,
        "rotation": page.rotation,
        "skew": page.skew,
        "lines": [{"box": ln.box, "text": ln.text, "conf": ln.conf} for ln in page.lines],
        "blocks": [{"label": b["label"], "bbox": r(b["bbox"], 1), "text": b["text"]} for b in page.blocks],
        "markdown": page.markdown,
        "text": page.text,
    }


def gray_pages(path: Path) -> list[np.ndarray]:
    return [g for _, g in P.load_pages(path)]


def run_case(case: str, path: Path, opts: dict) -> dict:
    conv = indra_ocr.load(path=str(MODELS), threads=1, **opts)
    recorder = Recorder(conv.ocr, case)
    for i, gray in enumerate(gray_pages(path)):
        cv2.imwrite(str(OUT / f"{path.stem}.p{i}.gray.png"), gray)
    with DocumentRecorder(recorder):
        pages = [page_json(page) for _, page in conv.convert(path)]
    # Every fixture is one page, so the events need no page key.
    assert len(pages) == 1, f"{case}: expected one page, got {len(pages)}"
    return {
        "case": case,
        "input": str(path.relative_to(ROOT)),
        "gray": [f"{path.stem}.p{i}.gray.png" for i in range(len(pages))],
        "options": opts,
        "indra_ocr": indra_ocr.__version__,
        "models": "datalab-to/indra-ocr@d7c839c892bfe4b8993373db64c62f154d05d70b",
        # cv2 resizes and warps through platform HALs (Carotene/KleidiCV on arm64,
        # IPP on x86): line crops and deskewed pages differ by platform.
        "platform": {"python": platform.platform(), "cv2": cv2.__version__, "hal": HAL},
        "runtime": {"onnxruntime": ort.__version__, "numpy": np.__version__},
        "events": recorder.events,
        "pages": pages,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for case, (path, opts) in CASES.items():
        data = run_case(case, path, opts)
        (OUT / f"{case}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
        print(f"{case}: {len(data['pages'])} page(s), {len(data['events'])} events")
    return 0


if __name__ == "__main__":
    sys.exit(main())
