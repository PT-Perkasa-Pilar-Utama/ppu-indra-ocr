"""Dump stage-by-stage goldens from the Python reference (indra-ocr 1.1.2).

Run by hand from the repo root:

    uvx --from indra-ocr==1.1.2 python -I scripts/golden/gen.py

It runs the reference on two model sets: models/v1.1/ (the v1.1 mirror, the
default preset) and models/ (the v3.1 mirror, the legacy preset). It reads the
fixtures in tests/fixtures/pages/ and assets/, and writes
tests/fixtures/golden/:

- <case>.json (v1.1) and v3.1/<case>.json: every stage the reference ran on
  the page, in call order, then the final Page fields.
- <input stem>.p<N>.gray.png: the grayscale page the OCR read (after decode or
  PDF render, before orient and deskew). TS stage tests start from these
  bytes, so JPEG decoders and PDF renderers cannot hide a pipeline difference.
- lm.<name>.json: log-probabilities of sampled (history, token) keys for each
  language model, for the TS n-gram reader.

CI never runs this script; the output is committed.
"""

from __future__ import annotations

import base64
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
from indra_ocr import lm as L
from indra_ocr import pipeline as P

ROOT = Path(__file__).resolve().parents[2]
HAL = next(
    (ln.split(":", 1)[1].strip() for ln in cv2.getBuildInformation().splitlines() if "Custom HAL" in ln),
    "none",
)
PAGES = ROOT / "tests" / "fixtures" / "pages"
OUT = ROOT / "tests" / "fixtures" / "golden"

# preset -> (model folder, output folder, Hugging Face commit)
MODEL_SETS = {
    "v1.1": (ROOT / "models" / "v1.1", OUT, "a1283c4b42a273c09480ad50e71461daff50fcc7"),
    "v3.1": (ROOT / "models", OUT / "v3.1", "d7c839c892bfe4b8993373db64c62f154d05d70b"),
}

# case name -> (input file, load() options)
LEGACY_CASES = {
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
CASES = {
    "v1.1": {
        **LEGACY_CASES,
        "multiscript": (PAGES / "multiscript.png", {}),
        "hebrew": (PAGES / "hebrew.png", {}),
        "vertical": (PAGES / "vertical.png", {}),
        "tilted-30": (PAGES / "tilted-30.png", {}),
        "sideways.auto": (PAGES / "sideways.pdf", {}),
        "rotate-90.auto": (PAGES / "rotate-90.pdf", {}),
        "rtl-text.auto": (PAGES / "rtl-text.pdf", {}),
        "vertical-text.auto": (PAGES / "vertical-text.pdf", {}),
        "legacy-font.auto": (PAGES / "legacy-font.pdf", {}),
    },
    "v3.1": LEGACY_CASES,
}

# Beam-search calls kept whole (inputs and result) for the TS beam search test.
BEAM_KEEP = 6


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


def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


class Recorder:
    """Wraps the stage methods of one OrtOCR and records each call."""

    def __init__(self, ocr: P.OrtOCR, case: str):
        self.ocr, self.case = ocr, case
        self.events: list[dict] = []
        self.beams: list[dict] = []
        self._wrap_sessions()
        self._wrap_methods()

    def log(self, stage: str, **data) -> None:
        self.events.append({"stage": stage, **data})

    def _wrap_sessions(self) -> None:
        rec = self
        ocr = self.ocr

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

        wrapped: dict[int, Session] = {}
        for name in ("det", "rec", "script", "order", "orient"):
            sess = getattr(ocr, name, None)
            if sess is not None:
                wrapped[id(sess)] = Session(name, sess)
                setattr(ocr, name, wrapped[id(sess)])
        for g, (sess, vocab) in list(ocr.groups.items()):
            wrapped[id(sess)] = Session(f"rec_{g}", sess)
            ocr.groups[g] = (wrapped[id(sess)], vocab)
        # The language models are found by their recognizer session.
        ocr.lms = {id(wrapped[k]): v for k, v in ocr.lms.items()}

    def on_run(self, name: str, feed: dict, outs: list) -> None:
        if name == "det":
            x = feed["x"]
            self.log("det.run", input_shape=list(x.shape), input_sha=sha(x), output_shape=list(outs[0].shape))
        elif name == "orient":
            self.log("orient.run", input_sha=sha(feed["x"]), p=r(outs[0][0], 6))
        elif name == "script":
            self.log("script.run", input_shape=list(feed["x"].shape), widths=r(feed["widths"]), p=r(outs[0], 6))
        elif name.startswith("rec"):
            self.log(
                f"{name}.run",
                input_shape=list(feed["x"].shape),
                input_sha=sha(feed["x"]),
                widths=r(feed["widths"]),
                lengths=r(outs[2]),
                lm_outputs=len(outs) > 3,
            )
        elif name == "order":
            outputs = {"order.a_shape": list(outs[0].shape)}
            if len(outs) >= 3:
                outputs["cls_argmax"] = r(outs[1][0].argmax(-1))
            self.log("order.run", feats=r(feed["feats"][0], 6), **outputs)

    def _wrap_methods(self) -> None:
        ocr, rec = self.ocr, self

        turn = ocr._turn

        def turn_(gray, k, a, small=None):
            out, k2, skew = turn(gray, k, a, small)
            rec.log("turn", k=k, a=r(a, 6), skew=r(skew, 6), shape=list(out.shape), sha=sha(out))
            return out, k2, skew

        ocr._turn = turn_

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

        line_groups = ocr.line_groups

        def line_groups_(crops, batch=64):
            names, sure = line_groups(crops, batch)
            rec.log("line_groups", names=names, sure=sure)
            return names, sure

        ocr.line_groups = line_groups_

        recognize = ocr.recognize

        def recognize_(gray, boxes, batch=64, vertical=False):
            out = recognize(gray, boxes, batch, vertical=vertical)
            lines = [{"text": t, "conf": r(c, 6), "toks": r(tk, 3)} for t, c, tk in out]
            rec.log("recognize", vertical=vertical, boxes=r(boxes[:, :4], 3), lines=lines)
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


class ModuleRecorder:
    """Wraps module functions of indra_ocr (pipeline, lm, document) for one case."""

    def __init__(self, rec: Recorder):
        self.rec = rec
        self.saved = [
            (P, "preview"),
            (P, "estimate_skew"),
            (P, "tidy_lines"),
            (L, "beam_search"),
            (D, "_render"),
            (D, "read_text_layer"),
            (D, "assign_runs"),
            (D, "readable"),
        ]
        self.fns = {(m.__name__, n): getattr(m, n) for m, n in self.saved}

    def __enter__(self):
        rec, fn = self.rec, self.fns

        def preview_(gray, side=P.PREVIEW_SIDE):
            out = fn["indra_ocr.pipeline", "preview"](gray, side)
            rec.log("preview", shape=list(out.shape), sha=sha(out))
            return out

        def estimate_skew_(gray, side=800, pool=None):
            a = fn["indra_ocr.pipeline", "estimate_skew"](gray, side, pool)
            rec.log("estimate_skew", shape=list(gray.shape), sha=sha(gray), angle=r(a, 6))
            return a

        def tidy_lines_(boxes, recs, rtl=False, space=None):
            texts_in = [t for t, _, _ in recs]
            b, out = fn["indra_ocr.pipeline", "tidy_lines"](boxes, recs, rtl, space)
            rec.log("tidy_lines", rtl=rtl, texts_in=texts_in, boxes=r(b, 3), texts=[t for t, _, _ in out])
            return b, out

        def beam_search_(blank, topv, topi, lm, alpha=0.5, beta=1.5, beam=8, blank_skip=0.999, cand_floor=1e-3):
            ids, steps = fn["indra_ocr.lm", "beam_search"](blank, topv, topi, lm, alpha, beta, beam, blank_skip, cand_floor)
            data = {"T": int(len(blank)), "alpha": alpha, "beta": beta, "beam": beam, "cand_floor": cand_floor, "ids": ids, "steps": steps}
            rec.log("beam_search", **data)
            if len(rec.beams) < BEAM_KEEP:
                rec.beams.append(
                    {
                        **data,
                        "lm": Path(getattr(lm, "path", "")).name,
                        "k": int(topv.shape[1]),
                        "blank": b64(blank, np.float32),
                        "topv": b64(topv, np.float32),
                        "topi": b64(topi, np.int64),
                    }
                )
            return ids, steps

        def render_(page, scale):
            out = fn["indra_ocr.document", "_render"](page, scale)
            rec.log("render", scale=r(scale, 8), shape=list(out.shape), sha=sha(out))
            return out

        def runs_json(lines):
            return [
                {"bbox": r(ln.bbox, 3), "text": ln.text, "quarter": ln.quarter, "seq": ln.seq, "chars": [[c, r(b, 3)] for c, b in ln.chars]}
                for ln in lines
            ]

        def read_text_layer_(page, gray, scale, turn=0, frame=None):
            tl = fn["indra_ocr.document", "read_text_layer"](page, gray, scale, turn, frame)
            rec.log(
                "read_text_layer",
                turn=turn,
                frame=frame,
                gray_shape=list(gray.shape),
                size=r(tl.size, 4),
                trusted=tl.trusted,
                reason=tl.reason,
                stats=r(tl.stats, 6),
                runs=runs_json(tl.lines),
                rotated=runs_json(tl.rotated),
            )
            return tl

        def assign_runs_(runs, boxes, k):
            toks, lost = fn["indra_ocr.document", "assign_runs"](runs, boxes, k)
            rec.log("assign_runs", k=r(k, 8), boxes=r(boxes[:, :4], 3), lost=lost, toks=[[[c, r(x, 3), i] for c, x, i in tk] for tk in toks])
            return toks, lost

        def readable_(ocr, runs):
            out = fn["indra_ocr.document", "readable"](ocr, runs)
            rec.log("readable", value=out)
            return out

        P.preview, P.estimate_skew, P.tidy_lines = preview_, estimate_skew_, tidy_lines_
        L.beam_search = beam_search_
        D._render, D.read_text_layer, D.assign_runs, D.readable = render_, read_text_layer_, assign_runs_, readable_
        return self

    def __exit__(self, *exc):
        for m, n in self.saved:
            setattr(m, n, self.fns[m.__name__, n])


def page_json(page) -> dict:
    return {
        "source": page.source,
        "note": page.note,
        "rotation": page.rotation,
        "skew": page.skew,
        "vertical": page.vertical,
        "lines": [{"box": ln.box, "text": ln.text, "conf": ln.conf} for ln in page.lines],
        "blocks": [{"label": b["label"], "bbox": r(b["bbox"], 1), "text": b["text"]} for b in page.blocks],
        "markdown": page.markdown,
        "text": page.text,
    }


def gray_pages(path: Path) -> list[np.ndarray]:
    return [g for _, g in P.load_pages(path)]


def run_case(case: str, path: Path, opts: dict, models: Path, commit: str) -> tuple[dict, list[dict]]:
    conv = indra_ocr.load(path=str(models), threads=1, **opts)
    # The LM file name tells the beam-search fixtures which model they used.
    for f, lm in [("lm.npz", conv.ocr.lm)] + [
        (f"lm_{g}.npz", conv.ocr.lms.get(id(s), (None,))[0]) for g, (s, _) in conv.ocr.groups.items()
    ]:
        if lm is not None:
            lm.path = f
    recorder = Recorder(conv.ocr, case)
    for i, gray in enumerate(gray_pages(path)):
        cv2.imwrite(str(OUT / f"{path.stem}.p{i}.gray.png"), gray)
    with ModuleRecorder(recorder):
        pages = [page_json(page) for _, page in conv.convert(path)]
    # Every fixture is one page, so the events need no page key.
    assert len(pages) == 1, f"{case}: expected one page, got {len(pages)}"
    data = {
        "case": case,
        "input": str(path.relative_to(ROOT)),
        "gray": [f"{path.stem}.p{i}.gray.png" for i in range(len(pages))],
        "options": opts,
        "indra_ocr": indra_ocr.__version__,
        "models": f"datalab-to/indra-ocr@{commit}",
        # cv2 resizes and warps through platform HALs (Carotene/KleidiCV on arm64,
        # IPP on x86): line crops and deskewed pages differ by platform.
        "platform": {"python": platform.platform(), "cv2": cv2.__version__, "hal": HAL},
        "runtime": {"onnxruntime": ort.__version__, "numpy": np.__version__},
        "events": recorder.events,
        "pages": pages,
    }
    return data, [{"case": case, **b} for b in recorder.beams]


def lm_samples(models: Path) -> None:
    """For each language model: keys seen in training at every history length,
    keys that back off, and unseen keys, with the reference's log-probability."""
    rng = np.random.default_rng(7)
    for f in sorted(models.glob("lm*.npz")):
        lm = L.NgramLM(f)
        keys: list[int] = []
        for n in range(lm.order):
            k = lm.k[n]
            pick = rng.choice(len(k), size=min(40, len(k)), replace=False)
            keys += [int(v) for v in k[pick]]
        # Random histories and tokens: mostly backoffs and unseen pairs.
        for _ in range(200):
            toks = rng.integers(-1, min(lm.vocab_size, 400), size=lm.order).tolist()
            keys.append(L._pack(toks, lm.bits))
        vals = lm.logp_keys(keys)
        # Python's per-key logp agrees with the vectorized one; record both paths' input.
        out = {
            "file": f.name,
            "order": lm.order,
            "bits": lm.bits,
            "vocab_size": lm.vocab_size,
            "max_logp": lm.max_logp,
            "keys": [str(k) for k in keys],
            "logp": vals,
        }
        (OUT / f"lm.{f.stem}.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"{f.name}: {len(keys)} keys")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for preset, (models, out, commit) in MODEL_SETS.items():
        out.mkdir(parents=True, exist_ok=True)
        beams: list[dict] = []
        for case, (path, opts) in CASES[preset].items():
            data, kept = run_case(case, path, opts, models, commit)
            beams += kept
            (out / f"{case}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
            print(f"{preset} {case}: {len(data['pages'])} page(s), {len(data['events'])} events")
        if beams:
            (out / "beam-search.json").write_text(json.dumps(beams, indent=1) + "\n")
    lm_samples(MODEL_SETS["v1.1"][0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
