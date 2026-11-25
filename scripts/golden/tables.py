"""Dump table goldens from the Python reference (indra-ocr 1.2.0): table_rel and table_verify.

Run by hand from the repo root:

    INDRA_OCR_NATIVE=0 uvx --from indra-ocr==1.2.0 python -I scripts/golden/tables.py

gen.py records the model inputs of the table stages but not what read_table and
TableVerifier.apply consume, so the TS port cannot replay them from it.  This script
runs the pages that hold tables (and the ones that hold none), wraps read_table and
TableVerifier.apply to keep their whole inputs and every intermediate, and adds
crafted cases for the pure functions and for the paths no page reaches: a
mirrored (right-to-left) table, the "geo" / "rel" column spans, a table read
without tsr_split and tsr_refine, a verifier relabelling, a decision that needs the
ruling features, a Table block without tsr_rel HTML, and the order-model table heads.

Writes tests/fixtures/golden/tables/:

- <case>.json: per page, the table_blocks calls (inputs, the recognizer's is_space and
  decode answers, every intermediate array) and the verifier calls.
- <case>.gray.png: the page table_blocks read (after turn and deskew), unless it
  equals the page PNG gen.py wrote; the JSON then names that file.
- units.json: crafted inputs of the pure functions, with the reference outputs.
- ruling.png: the page of the ruling() cases of units.json.
- verify-crafted.json: TableVerifier.apply on small pages built to reach each decision path, and with
  the order model's table heads; verify-*.png are those pages.
- <case>.json also holds "variants" (table.png read other ways) and "verify_forced" (a verifier with
  threshold 0, which relabels the table and drops its tsr_rel HTML).

CI never runs this script; the output is committed.  Arrays are base64 of
little-endian float32 (int64 for word ids) in {"shape", "b64"} objects.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sys
import zlib
from pathlib import Path

if os.environ.get("INDRA_OCR_NATIVE") != "0":
    sys.exit("run with INDRA_OCR_NATIVE=0 (the TS port matches the Python clustering, not the C++ kernel)")

import cv2
import numpy as np

import indra_ocr
from indra_ocr import layout_post as LP
from indra_ocr import pipeline as P
from indra_ocr import table_rel as T
from indra_ocr import table_verify as V

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "tests" / "fixtures" / "pages"
GOLDEN = ROOT / "tests" / "fixtures" / "golden"
OUT = GOLDEN / "tables"
MODELS = ROOT / "models" / "v1.2"

CASES = {
    "table": PAGES / "table.png",
    "receipt": ROOT / "assets" / "receipt.jpg",
    "digital": PAGES / "digital.pdf",
    "sideways": PAGES / "sideways.pdf",
    "scanned": PAGES / "scanned.pdf",
}
# gen.py's page PNG of each input (before orient and deskew): reused when table_blocks read the same pixels.
GEN_PNG = {"table": "table.p0.gray.png", "receipt": "receipt.p0.gray.png", "digital": "digital.p0.gray.png",
           "sideways": "sideways.p0.gray.png", "scanned": "scanned.p0.gray.png"}


def arr(a, dtype=np.float32) -> dict:
    a = np.ascontiguousarray(a, dtype=dtype)
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def jsonable(x):
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(jsonable(data), ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1024:.0f} KiB")


def key_of(blk) -> str:
    return ",".join(map(str, blk))


# ---------------------------------------------------------------- read_table capture

CUR: dict | None = None
ORIG = {n: getattr(T, n) for n in ("words_of", "word_features", "split_maps", "split_features", "norm_boxes",
                                    "word_ids", "refine_text_features", "decode", "table_html")}


class Spy:
    """A session whose run() keeps its feeds and outputs in the read being captured."""

    def __init__(self, name: str, sess):
        self.name, self.sess = name, sess

    def __getattr__(self, attr):
        return getattr(self.sess, attr)

    def run(self, names, feed):
        outs = self.sess.run(names, feed)
        got = names or [o.name for o in self.sess.get_outputs()]
        o = dict(zip(got, outs))
        if self.name == "rel":
            CUR["model_in"] = {"feat": arr(feed["feat"]), "box": arr(feed["box"]), "wid": arr(feed["wid"], np.int64) if "wid" in feed else None}
            CUR["model_out"] = {"pair": arr(o["pair"]), "tag": arr(o["tag"])}
        elif self.name == "split":
            CUR["split_in"] = {"shape": list(feed["x"].shape), "sha": sha(feed["x"])}
        else:
            CUR["refine_in"] = {"tfeat": arr(feed["tfeat"])}
            CUR["refine_out"] = {"tag_out": arr(o["tag_out"])}
        return outs


def _words_json(words) -> list:
    return [{"box": w["box"], "text": w["text"], "line": w["line"], "pos": w["pos"], "n": w["n"]} for w in words]


def _wrap_rel_functions() -> None:
    def words_of(lines, is_space, decode):
        out = ORIG["words_of"](lines, is_space, decode)
        CUR["words"] = _words_json(out)
        return out

    def word_features(words, gray, bbox):
        f = ORIG["word_features"](words, gray, bbox)
        CUR["wf"] = {"words": _words_json(words), "bbox": [float(v) for v in bbox], "feat": arr(f)}
        return f

    def split_maps(sess, gray, bbox, line_h):
        maps = ORIG["split_maps"](sess, gray, bbox, line_h)
        CUR["maps"] = {"bbox": [float(v) for v in bbox], "line_h": float(line_h),
                       "result": None if maps is None else {"shape": list(maps[0].shape), "s": float(maps[1]), "origin": list(maps[2]), "sha": sha(maps[0])}}
        return maps

    def split_features(boxes, maps):
        f = ORIG["split_features"](boxes, maps)
        CUR["split_feat"] = arr(f)
        return f

    def norm_boxes(boxes):
        out = ORIG["norm_boxes"](boxes)
        CUR["norm_boxes"] = {"boxes": arr(boxes), "out": arr(out)}
        return out

    def word_ids(texts, n_vocab):
        out = ORIG["word_ids"](texts, n_vocab)
        CUR["word_ids"] = {"n_vocab": n_vocab, "ids": out.tolist()}
        return out

    def refine_text_features(texts):
        out = ORIG["refine_text_features"](texts)
        CUR["tfeat"] = arr(out)
        return out

    def decode(pair, tags, boxes, col_spans="geo", texts=None):
        out = ORIG["decode"](pair, tags, boxes, col_spans, texts)
        CUR["decode"] = {"pair": arr(pair), "tags": arr(tags), "boxes": arr(boxes), "col_spans": col_spans, "texts": texts, "struct": out}
        return out

    def table_html(struct, words):
        out = ORIG["table_html"](struct, words)
        CUR["html"] = out
        return out

    for name, fn in locals().items():
        if name in ORIG:
            setattr(T, name, fn)


def _dedupe(rec: dict) -> None:
    """Drop arrays the record holds twice (the model inputs are the feature, box and word-id records)."""
    if "model_in" not in rec:
        return
    mi, dec = rec["model_in"], rec["decode"]
    assert mi["box"]["b64"] == rec["norm_boxes"]["out"]["b64"] and dec["boxes"]["b64"] == rec["norm_boxes"]["boxes"]["b64"]
    nf = mi["feat"]["shape"][-1]
    base = np.frombuffer(base64.b64decode(rec["wf"]["feat"]["b64"]), np.float32).reshape(rec["wf"]["feat"]["shape"])
    full = np.frombuffer(base64.b64decode(mi["feat"]["b64"]), np.float32).reshape(mi["feat"]["shape"])[0]
    expect = base if nf <= base.shape[1] else np.concatenate([base, np.frombuffer(base64.b64decode(rec["split_feat"]["b64"]), np.float32).reshape(rec["split_feat"]["shape"])], 1)
    assert np.array_equal(full, expect[:, :nf]), "model feat is not wf + split"
    rec["model_in"] = {"n_feat": int(nf), "wid": mi["wid"]}
    del dec["boxes"]
    pair = np.frombuffer(base64.b64decode(dec["pair"]["b64"]), np.float32).reshape(dec["pair"]["shape"])
    out = np.frombuffer(base64.b64decode(rec["model_out"]["pair"]["b64"]), np.float32).reshape(rec["model_out"]["pair"]["shape"])
    assert np.array_equal(pair, out[0])
    del dec["pair"]


def capture_read(sess, gray, lines, is_space, decode_text, mirror=False, col_spans="geo", split_sess=None, refine_sess=None) -> tuple[str | None, dict]:
    """T.read_table with its inputs and intermediates kept.  Returns (html, record)."""
    global CUR
    spaces: list = []
    decodes: list = []

    def sp(tok):
        v = is_space(tok)
        spaces.append(bool(v))
        return v

    def dec(toks):
        text = decode_text(toks)
        decodes.append({"toks": [[jsonable(t[0]), float(t[1])] for t in toks], "text": text})
        return text

    CUR = {}
    html = T.read_table(Spy("rel", sess), gray, lines, sp, dec, mirror, col_spans,
                        None if split_sess is None else Spy("split", split_sess),
                        None if refine_sess is None else Spy("refine", refine_sess))
    rec = CUR
    CUR = None
    _dedupe(rec)
    rec["args"] = {
        "lines": [{"box": [float(v) for v in b], "toks": [[jsonable(t[0]), float(t[1])] for t in toks], "space": None} for b, toks in lines],
        "mirror": mirror, "col_spans": col_spans, "split": split_sess is not None, "refine": refine_sess is not None,
    }
    # is_space answered once per token, line by line, in order.
    k = 0
    for ln in rec["args"]["lines"]:
        ln["space"] = spaces[k : k + len(ln["toks"])]
        k += len(ln["toks"])
    assert k == len(spaces)
    rec["decodes"] = decodes
    rec["html"] = html
    return html, rec


# ---------------------------------------------------------------- verifier capture

VCUR: dict | None = None
V_ORIG = {"block_features": V.block_features, "ruling": V.ruling, "decided": V.TableVerifier.decided,
          "p_false": V.TableVerifier.p_false, "apply": V.TableVerifier.apply,
          "block_part": LP._block_part}


def _wrap_verify_functions() -> None:
    def block_features(blk, boxes, texts, cls_logits, html_table, gray, n_tables, heads_ok, with_ruling=True):
        f = V_ORIG["block_features"](blk, boxes, texts, cls_logits, html_table, gray, n_tables, heads_ok, with_ruling)
        VCUR["blocks"].append({"key": key_of(blk), "html": html_table, "n_tables": n_tables, "heads_ok": bool(heads_ok), "features": f, "decided": None, "rule": None, "p_calls": []})
        return f

    def ruling(gray, box):
        out = V_ORIG["ruling"](gray, box)
        if VCUR is None:
            return out
        VCUR["blocks"][-1]["rule"] = [float(v) for v in out]
        VCUR["blocks"][-1]["rule_box"] = [float(v) for v in box]
        return out

    def decided(self, f):
        out = V_ORIG["decided"](self, f)
        VCUR["blocks"][-1]["decided"] = out
        VCUR["blocks"][-1]["p_calls"] = []  # decided() probes the corners of the ruling features; only the later call counts
        return out

    def p_false(self, f):
        out = V_ORIG["p_false"](self, f)
        VCUR["blocks"][-1]["p_calls"].append([out, bool(out >= self.threshold)])
        return out

    def block_part(parts, blk, boxes, texts, cls_logits, keep_furniture, cells, tab):
        n = len(parts)
        V_ORIG["block_part"](parts, blk, boxes, texts, cls_logits, keep_furniture, cells, tab)
        if VCUR is None:  # page_markdown calls it too
            return
        VCUR["parts"][key_of(blk)] = parts[n:]

    V.block_features, V.ruling = block_features, ruling
    V.TableVerifier.decided, V.TableVerifier.p_false = decided, p_false
    LP._block_part = block_part


def _cells_json(cells) -> dict | None:
    if cells is None:
        return None
    return {str(i): [[[float(v) for v in bx], tx] for bx, tx in cs] for i, cs in cells.items()}


def capture_apply(verifier, weights, boxes, texts, order, cls, same, tab, cells, tables, gray, gray_file) -> tuple:
    global VCUR
    VCUR = {"blocks": [], "parts": {}}
    inp = {
        "weights": weights, "gray": gray_file, "boxes": arr(boxes), "texts": texts, "order": [int(i) for i in order],
        "cls": arr(cls), "same": arr(same), "tab": None if tab is None else arr(tab), "cells": _cells_json(cells),
        "tables": None if tables is None else {key_of(k): v for k, v in tables.items() if k != "ms"},
    }
    out_cls, out_tab = V_ORIG["apply"](verifier, boxes, texts, order, cls, same, tab, cells, tables, gray)
    rec = {**inp, "out": {"relabelled": out_cls is not cls, "cls": arr(out_cls),
                          "tables": None if out_tab is None else {key_of(k): v for k, v in out_tab.items() if k != "ms"}},
           "blocks": VCUR["blocks"], "parts": VCUR["parts"]}
    VCUR = None
    return (out_cls, out_tab), rec


# ---------------------------------------------------------------- page runs


class PageCapture:
    """The table_blocks and verifier.apply calls of one page."""

    def __init__(self, name: str, gen_png: str):
        self.name, self.gen_png = name, gen_png
        self.table_calls: list[dict] = []
        self.verify_calls: list[dict] = []
        self.gray_file: str | None = None
        self._gray_sha: str | None = None
        self.raw: dict = {}

    def gray_ref(self, gray: np.ndarray) -> str:
        s = sha(gray)
        if s == self._gray_sha:
            return self.gray_file
        assert self._gray_sha is None, "two different pages in one case"
        ref = cv2.imread(str(GOLDEN / self.gen_png), cv2.IMREAD_GRAYSCALE)
        if ref is not None and ref.shape == gray.shape and np.array_equal(ref, gray):
            self.gray_file = self.gen_png
        else:
            self.gray_file = f"tables/{self.name}.gray.png"
            cv2.imwrite(str(GOLDEN / self.gray_file), gray)
        self._gray_sha = s
        return self.gray_file


def run_case(name: str, path: Path) -> None:
    conv = indra_ocr.load(path=str(MODELS), threads=1)
    ocr = conv.ocr
    cap = PageCapture(name, GEN_PNG[name])
    state: dict = {}

    table_blocks = P.OrtOCR.table_blocks
    real_read = T.read_table

    def read_table(sess, gray, lines, is_space, decode_text, mirror=False, col_spans="geo", split_sess=None, refine_sess=None):
        html, rec = capture_read(sess, gray, lines, is_space, decode_text, mirror, col_spans, split_sess, refine_sess)
        k = len(state["calls"])
        rec["key"] = state["keys"][k]
        state["calls"].append(rec)
        return html

    def table_blocks_(self, gray, boxes, rec, order, layout, geo, from_layer=False):
        state["keys"] = [key_of(b) for b in LP.split_blocks(order, layout[1]) if len(b) >= 2 and int(layout[0][b].sum(0).argmax()) == LP.TABLE]
        state["calls"] = []
        P.read_table = read_table
        try:
            out = table_blocks(self, gray, boxes, rec, order, layout, geo, from_layer)
        finally:
            P.read_table = real_read
        gref = cap.gray_ref(gray)
        cap.table_calls.append({"from_layer": from_layer, "gray": gref, "page_rtl": bool(self._page_rtl),
                                "reads": state["calls"], "tables": {key_of(k): v for k, v in out.items() if k != "ms"}})
        return out

    def apply_(self, boxes, texts, order, cls, same, tab, cells, tables, gray):
        cap.raw.setdefault("apply", (boxes, texts, order, cls, same, tab, cells, tables, gray))
        weights = "layer" if self is ocr.table_verify_layer else "ocr"
        ret, rec = capture_apply(self, weights, boxes, texts, order, cls, same, tab, cells, tables, gray, cap.gray_ref(gray))
        cap.verify_calls.append(rec)
        return ret

    P.OrtOCR.table_blocks = table_blocks_
    V.TableVerifier.apply = apply_
    try:
        pages = [page for _, page in conv.convert(path)]
    finally:
        P.OrtOCR.table_blocks = table_blocks
        V.TableVerifier.apply = V_ORIG["apply"]
    assert len(pages) == 1 and pages[0].source != "error", pages[0].note
    n_reads = sum(len(c["reads"]) for c in cap.table_calls)
    n_blocks = sum(len(v["blocks"]) for v in cap.verify_calls)
    print(f"{name}: {n_reads} read_table calls, {n_blocks} verified blocks")
    if not n_reads and not n_blocks:
        return
    data = {"case": name, "input": str(path.relative_to(ROOT)), "indra_ocr": indra_ocr.__version__, "numpy": np.__version__,
            "tsr_rel_span": ocr.tsr_rel_span, "tables": cap.table_calls, "verify": cap.verify_calls}
    if name == "table":
        # A verifier that calls every table false: the relabelling path with a tsr_rel table to drop.
        weights = json.loads((MODELS / "table_verify.json").read_text())
        boxes_, texts_, order_, cls_, same_, tab_, cells_, tables_, gray_ = cap.raw["apply"]
        forced = V.TableVerifier({**weights, "threshold": 0.0})
        (_, _), rec = capture_apply(forced, "ocr", boxes_, texts_, order_, cls_, same_, tab_, cells_, tables_, gray_, cap.gray_file)
        rec["threshold"] = 0.0
        data["verify_forced"] = [rec]
    # Variants of the first read: what no page reaches (mirror, other column spans, no split / refine model).
    if name == "table" and cap.table_calls and cap.table_calls[0]["reads"]:
        first = cap.table_calls[0]["reads"][0]
        gray = cv2.imread(str(GOLDEN / cap.gray_file), cv2.IMREAD_GRAYSCALE)
        lines = [(np.array(ln["box"], np.float32), [tuple(t) for t in ln["toks"]]) for ln in first["args"]["lines"]]
        by_tok = {json.dumps(d["toks"]): d["text"] for d in first["decodes"]}
        all_spaces = [s for ln in first["args"]["lines"] for s in ln["space"]]
        decode_text = lambda toks: by_tok[json.dumps([[jsonable(t[0]), float(t[1])] for t in toks])]  # noqa: E731
        variants = []
        for label, kw in (("rel", {"col_spans": "rel"}), ("geo", {"col_spans": "geo"}), ("rel2 mirror", {"col_spans": "rel2", "mirror": True}),
                          ("geo mirror", {"col_spans": "geo", "mirror": True}),
                          ("rel2 without split", {"col_spans": "rel2", "split_sess": None, "refine_sess": ocr.tsr_refine}),
                          ("rel2 without refine", {"col_spans": "rel2", "split_sess": ocr.tsr_split, "refine_sess": None}),
                          ("rel2 without both", {"col_spans": "rel2", "split_sess": None, "refine_sess": None})):
            kw = {"split_sess": ocr.tsr_split, "refine_sess": ocr.tsr_refine, **kw}
            answers = iter(all_spaces)  # words_of asks once per token, in line order
            _, rec = capture_read(ocr.tsr_rel, gray, lines, lambda tok: next(answers), decode_text, **kw)
            # the variants test readTable end to end: inputs, the decode input and the HTML
            keep = ("args", "decodes", "html", "decode")
            variants.append({"label": label, **{k: rec[k] for k in keep}})
        data["variants"] = variants
    dump(OUT / f"{name}.json", data)


# ---------------------------------------------------------------- crafted cases


def crafted_units() -> dict:
    rng = np.random.default_rng(11)
    units: dict = {}

    # decode on a synthetic grid: 4 rows x 3 columns, a header over columns 1-2, a lone section row, an outside word
    def grid_case(seed: int, with_out: bool, stub: bool = False):
        r = np.random.default_rng(seed)
        words, cell_of, row_of, col_of = [], [], [], []
        x = [20.0, 120.0, 220.0]
        for ri in range(4):
            for ci in range(3):
                words.append([x[ci] + r.uniform(0, 3), 30.0 + 20 * ri, x[ci] + 60 + r.uniform(0, 3), 42.0 + 20 * ri])
                cell_of.append(len(words) - 1); row_of.append(ri); col_of.append(ci)
        words.append([20.0, 130.0, 280.0, 142.0]); cell_of.append(len(words) - 1); row_of.append(4); col_of.append(0)
        # a two-word cell: the second word right after the first one of (1, 1)
        words.append([184.0, 50.0, 200.0, 62.0]); cell_of.append(cell_of[4]); row_of.append(1); col_of.append(1)
        if with_out:
            words.append([20.0, 5.0, 150.0, 17.0]); cell_of.append(len(words) - 1); row_of.append(-1); col_of.append(-1)
        n = len(words)
        boxes = np.array(words, np.float32)
        pair = np.zeros((3, n, n), np.float32)
        for i in range(n):
            for j in range(n):
                same = [cell_of[i] == cell_of[j], row_of[i] == row_of[j] and row_of[i] >= 0, col_of[i] == col_of[j] and col_of[i] >= 0]
                for k in range(3):
                    pair[k, i, j] = (4.0 if same[k] else -4.0) + r.normal(0, 1.2)
        if stub:
            # a stub header: the first word is as tall as two rows and shares a row with both
            boxes[0, 3] = boxes[3, 3]
            for j in range(n):
                if row_of[j] in (0, 1):
                    pair[1, 0, j] = pair[1, j, 0] = 4.0
        pair = ((pair + pair.transpose(0, 2, 1)) / 2).astype(np.float32)
        tags = np.zeros((n, 2), np.float32)
        tags[:, 0] = np.where(np.array(row_of) <= (1 if stub else 0), 3.0, -3.0) + r.normal(0, 0.5, n)
        tags[:, 1] = -3.0 + r.normal(0, 0.5, n)
        if with_out:
            tags[-1, 1] = 3.0
        texts = [f"w{i}" if i % 3 else f"Name {i}" for i in range(n)]
        texts[12] = "Section title"
        return pair, tags, boxes, texts

    cases = []
    for seed, out, stub in ((1, False, False), (2, True, False), (3, True, False), (4, False, True)):
        pair, tags, boxes, texts = grid_case(seed, out, stub)
        for cs in ("geo", "rel", "rel2"):
            struct = ORIG["decode"](pair, tags, boxes, cs, texts)
            cases.append({"pair": arr(pair), "tags": arr(tags), "boxes": arr(boxes), "col_spans": cs, "texts": texts, "struct": struct})
        cases.append({"pair": arr(pair), "tags": arr(tags), "boxes": arr(boxes), "col_spans": "geo", "texts": None,
                      "struct": ORIG["decode"](pair, tags, boxes, "geo", None)})
    # too few kept words, and no words
    pair, tags, boxes, texts = grid_case(1, False)
    tags2 = tags.copy(); tags2[:, 1] = 3.0
    cases.append({"pair": arr(pair), "tags": arr(tags2), "boxes": arr(boxes), "col_spans": "geo", "texts": texts, "struct": ORIG["decode"](pair, tags2, boxes, "geo", texts)})
    units["decode"] = cases

    # table_html on structs: overlapping spans, outside words, leader dots, markup characters
    words = [{"box": [float(10 * i), float(20 * (i // 3)), float(10 * i + 8), float(20 * (i // 3) + 10)], "text": t, "line": i // 3, "pos": i % 3, "n": 3}
             for i, t in enumerate(["Item", "Qty", "Price", "Tom & Jerry", "<b>", "x > y", "Total . . .", "3", "$ 4.50 \u2026", "note", "a\u00b7b", "tail . .", "z"])]
    structs = [
        {"cells": [(0, 0, 0, 0, [0]), (0, 0, 1, 2, [1, 2]), (1, 1, 0, 0, [3]), (1, 1, 1, 1, [4]), (1, 1, 2, 2, [5])], "n_head": 1, "outside": [], "n_rows": 2, "n_cols": 3},
        {"cells": [(0, 1, 0, 0, [0]), (0, 0, 1, 2, [1, 2]), (1, 2, 1, 2, [3]), (2, 2, 0, 0, [4])], "n_head": 1, "outside": [12, 11], "n_rows": 3, "n_cols": 3},
        {"cells": [(0, 0, 0, 1, [6, 7]), (0, 0, 1, 1, [8]), (1, 1, 0, 0, [9, 10])], "n_head": 0, "outside": [], "n_rows": 2, "n_cols": 2},
        {"cells": [(0, 0, 0, 0, [0])], "n_head": 0, "outside": [], "n_rows": 1, "n_cols": 1},
        {"cells": [(0, 0, 0, 0, [0, 6]), (0, 0, 1, 1, [11, 8])], "n_head": 1, "outside": [], "n_rows": 1, "n_cols": 2},
        {"cells": [(1, 1, 0, 0, [0]), (0, 1, 0, 1, [1, 2])], "n_head": 1, "outside": [], "n_rows": 2, "n_cols": 2},
    ]
    units["table_html"] = {"words": words, "cases": [{"struct": s, "html": ORIG["table_html"](s, words)} for s in structs]}

    texts = ["Total", "TOTAL", "12,345.67", "98765", "\u0661\u0662\u0663", "\uff46\uff55\uff4c\uff4c", "na\u00efve", "\u041f\u0440\u0438\u0432\u0435\u0442", "\u65e5\u672c\u8a9e",
             "\u01c5", "\u02b0ello", "x\u00b2", "\u0130stanbul", "\u00df", "\u212a", "", "a-b", "$12", "\u00e9t\u00e9", "\u03a3\u03af\u03c3\u03c5\u03c6\u03bf\u03c2", "ab\u0301", "\u2160", "\ua7b4", "\u1d00bc", "\u0278x", "\u02e0a"]
    units["word_ids"] = {"n_vocab": 4096, "texts": texts, "ids": [int(v) for v in ORIG["word_ids"](texts, 4096)],
                         "crc32": [{"s": s, "crc": zlib.crc32(s.encode("utf-8"))} for s in ("", "a", "0", "total", "na\u00efve", "\u00e9t\u00e9 0.00")]}
    # every code point whose letter-ness the port must tell: which isalpha characters have a name that starts with LATIN
    import unicodedata
    latin = []
    start = None
    for cp in range(0x110000):
        c = chr(cp)
        ok = c.isalpha() and unicodedata.name(c, "").startswith("LATIN")
        if ok and start is None:
            start = cp
        if not ok and start is not None:
            latin.append([start, cp - 1])
            start = None
    units["latin_ranges"] = latin
    digits = []
    start = None
    for cp in range(0x110000):
        ok = chr(cp).isdigit()
        if ok and start is None:
            start = cp
        if not ok and start is not None:
            digits.append([start, cp - 1])
            start = None
    units["digit_ranges"] = digits

    rtexts = ["", "12", "1,234.5", "(3.2)", "-4", "\u22125%", "$ 1,200", "$(12)", "n/a", "N/A", "-", "\u2013", "\u2026", "...", "\u00b7", "*", "**", "\u2020", "#", "a", "A", "Z", "Q1", "FY2024", "Net sales:", "Total revenue", "AB", "Ab", "ab",
              "\u20ac5", "[1]", "50%", "12*", "1 2", "\u00a0 7 ", "\u0663", "\uff11\uff12", "x\u00b2", "Rev.", "nd", "NR", "\uff0a", "\u00bd", "(", "9)", "1.2.3", "1e5"]
    units["refine_text_features"] = {"texts": rtexts, "out": arr(ORIG["refine_text_features"](rtexts))}

    pt = [
        "<table>\n<tr><th>A &amp; B</th><th colspan=\"2\">C</th></tr>\n<tr><td>1</td><td>x &lt; y</td><td></td></tr>\n</table>",
        "<table><tr><td rowspan=\"2\">1</td><td colspan=\"3\">wide</td><td colspan=\"1\">z</td></tr><tr><td> sp </td></tr></table>",
        "<table><tr><td><b>bold</b> text</td><td>&#65;&#x42;&#x1F600;</td><td>&gt;&gt;&amp;lt;</td></tr></table>",
        "Title text\n\n<table><tr><th>Only</th></tr></table>",
        "<table></table>", "no table here", "<TR><TD>upper</TD></TR>",
    ]
    import html as html_mod

    unescape = ["&#128; &#x9f; &#0; &#13; &#1; &#xFFFE; &#xD800; &#x110000; &amp &ampx; &lt&quot &nbsp; &apos; &#65 &#x41 &#xZZ; &unknown;",
                "a & b", "&&amp;", "&#9999999999;", "&lt;&gt;&amp;lt;", "&#x1F600;&#X41;&#xb;&#xe;&#x7f;&#xfdd0;", "no entity", "&", "&#;", "&#x;"]
    units["html_unescape"] = [{"s": t, "out": html_mod.unescape(t)} for t in unescape]
    units["parse_table"] = [{"html": h, "rows": V.parse_table(h)} for h in pt]

    runs = []
    for h, w, length in ((3, 12, 4), (3, 12, 12), (3, 12, 13), (4, 9, 1), (4, 9, 2), (5, 20, 8), (2, 5, 8), (6, 17, 7), (1, 1, 1), (3, 30, 11), (4, 8, 3), (4, 8, 8)):
        for density in (0.5, 0.8, 0.95):
            ink = rng.random((h, w)) < density
            if h > 2:
                ink[1] = True
            if h > 3:
                ink[2, : w // 2] = True
            runs.append({"h": h, "w": w, "length": length, "ink": ink.astype(np.uint8).ravel().tolist(), "out": [bool(v) for v in V._any_run(ink, length)]})
    units["any_run"] = runs

    aligns = []
    for n in (1, 2, 3, 7, 12):
        b = np.zeros((n, 4), np.float32)
        for i in range(n):
            y = 20.0 * (i // 3) + rng.uniform(-2, 2)
            x = 30.0 * (i % 3) + rng.uniform(0, 8)
            b[i] = [x, y, x + rng.uniform(20, 70), y + rng.uniform(9, 13)]
        aligns.append({"boxes": arr(b), "out": V.alignment(b)})
    b = np.array([[0, 0, 50, 10], [0, 0, 50, 10], [0, 0, 50, 10]], np.float32)
    aligns.append({"boxes": arr(b), "out": V.alignment(b)})
    b = np.array([[0, 10, 50, 10], [60, 10, 90, 10]], np.float32)
    aligns.append({"boxes": arr(b), "out": V.alignment(b)})
    units["alignment"] = aligns

    # ruling: a page with a few rules and boxes over and away from them
    g = np.full((200, 300), 235, np.uint8)
    cv2.line(g, (20, 40), (280, 40), 20, 2)
    cv2.line(g, (20, 90), (280, 90), 20, 1)
    cv2.line(g, (60, 30), (60, 120), 30, 2)
    cv2.putText(g, "Hello", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    cv2.imwrite(str(OUT / "ruling.png"), g)
    rul = []
    for box in ((10, 30, 290, 130), (10, 130, 290, 190), (100, 100, 200, 130), (0, 0, 3, 3), (10.5, 30.5, 150.25, 100.75), (250, 150, 400, 300), (-5, -5, 100, 50)):
        rul.append({"box": list(box), "out": list(V_ORIG["ruling"](g, np.array(box, np.float32)))})
    units["ruling"] = {"gray": "tables/ruling.png", "cases": rul}
    return units


def synthetic_verify(verifier_ocr, verifier_layer) -> list[dict]:
    """TableVerifier.apply on small pages built to reach each decision path."""
    out = []
    r = np.random.default_rng(5)

    def page(kind: str, trial: int):
        """A page with one Table block: (boxes, texts, cls, same, order, cells, tables, gray)."""
        if kind == "paragraph":
            n = int(r.integers(3, 9))
            boxes = [[40.0, 30.0 + 16 * i, float(300 + r.integers(-20, 20)), 42.0 + 16 * i] for i in range(n)]
            texts = [f"This is sentence number {i} of a paragraph that goes on." for i in range(n)]
        elif kind == "formulas":
            n = int(r.integers(2, 6))
            boxes = [[100.0, 30.0 + 22 * i, 220.0 + r.integers(0, 30), 46.0 + 22 * i] for i in range(n)]
            texts = [f"$x_{i} = a + b^2$" for i in range(n)]
        else:
            nr, nc = int(r.integers(3, 8)), int(r.integers(2, 5))
            boxes, texts = [], []
            for i in range(nr):
                for j in range(nc):
                    x = 40.0 + 70 * j + r.uniform(0, 3)
                    boxes.append([x, 30.0 + 16 * i, x + 50, 42.0 + 16 * i])
                    texts.append(["Item", "1,234", "5.6%", "$ 77", "Name:", "n/a"][int(r.integers(0, 6))] if i else f"Col {j}")
            n = len(boxes)
        n = len(boxes)
        boxes = np.array(boxes, np.float32)
        cls = np.full((n, 10), -4.0, np.float32)
        cls[:, LP.TABLE] = 4.0
        same = np.full((n, n), 5.0, np.float32)
        gray = np.full((220, 360), 240, np.uint8)
        # rules: a share of the trials get long horizontal and vertical strokes over the block
        if trial % 3 == 0:
            for i in range(0, 8):
                cv2.line(gray, (30, 28 + 16 * i), (340, 28 + 16 * i), 40, 1)
        if trial % 4 == 0:
            for j in range(0, 5):
                cv2.line(gray, (38 + 70 * j, 20), (38 + 70 * j, 200), 40, 1)
        cells = {i: [([float(v) for v in boxes[i]], texts[i])] for i in range(n)}
        return boxes, texts, cls, same, list(range(n)), cells, gray

    want = {"relabel": 2, "keep": 2, "ruling_relabel": 1, "ruling_keep": 1}
    got = {k: 0 for k in want}
    for trial in range(400):
        if all(got[k] >= want[k] for k in want):
            break
        kind = ("paragraph", "formulas", "grid")[trial % 3]
        boxes, texts, cls, same, order, cells, gray = page(kind, trial)
        tables = None
        if kind == "grid" and trial % 2:
            tables = {tuple(range(len(boxes))): "<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>"}
        for name, ver in (("ocr", verifier_ocr), ("layer", verifier_layer)):
            if ver is None:
                continue
            # classify the path with the reference itself
            (_, _), rec = capture_apply(ver, name, boxes, texts, order, cls, same, None, cells, tables, gray, None)
            b = rec["blocks"][0]
            if b["decided"] is None:
                label = "ruling_relabel" if b["p_calls"][-1][1] else "ruling_keep"
            else:
                label = "relabel" if b["decided"] else "keep"
            if got[label] >= want[label]:
                continue
            got[label] += 1
            fname = f"verify-{label}-{name}-{trial}.png"
            cv2.imwrite(str(OUT / fname), gray)
            rec["gray"] = f"tables/{fname}"
            rec["label"] = label
            rec["kind"] = kind
            out.append(rec)
    print("synthetic verifier paths:", got)
    return out


def tab_cases() -> list[dict]:
    """The order model's table heads: heads_plausible, table_html_heads and apply with `tab`."""
    out = []
    r = np.random.default_rng(9)
    for stacked in (True, False):
        # 2 columns x 3 rows of lines; cell = a line, or (stacked) two stacked lines
        n = 6
        boxes = np.array([[30.0 + 100 * (i % 2), 30.0 + 18 * (i // 2), 110.0 + 100 * (i % 2), 42.0 + 18 * (i // 2)] for i in range(n)], np.float32)
        texts = [f"cell {i}" for i in range(n)]
        tab = np.full((3, n, n), -3.0, np.float32)
        for i in range(n):
            for j in range(n):
                if i // 2 == j // 2:
                    tab[0, i, j] = 3.0
                if i % 2 == j % 2:
                    tab[1, i, j] = 3.0
                if i == j:
                    tab[2, i, j] = 3.0
        if stacked:
            tab[2, 0, 2] = tab[2, 2, 0] = 3.0  # one cell of two stacked lines
        else:
            tab[2, 0, 1] = tab[2, 1, 0] = 3.0  # side-by-side lines in one cell: heads fail
            tab[2, 2, 3] = tab[2, 3, 2] = 3.0
        tab += r.normal(0, 0.1, tab.shape).astype(np.float32)
        out.append({"boxes": arr(boxes), "texts": texts, "tab": arr(tab), "plausible": LP.heads_plausible(boxes, tab),
                    "html": LP.table_html_heads(boxes, texts, tab)})
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _wrap_rel_functions()
    _wrap_verify_functions()
    for name, path in CASES.items():
        run_case(name, path)
    ocr_v, layer_v = V.TableVerifier.load(MODELS / "table_verify.json")
    units = crafted_units()
    dump(OUT / "units.json", units)
    synth = synthetic_verify(ocr_v, layer_v)
    # tab cases run through apply too
    tabs = tab_cases()
    tab_apply = []
    for t in tabs:
        boxes = np.frombuffer(base64.b64decode(t["boxes"]["b64"]), np.float32).reshape(t["boxes"]["shape"])
        tab = np.frombuffer(base64.b64decode(t["tab"]["b64"]), np.float32).reshape(t["tab"]["shape"])
        n = len(boxes)
        cls = np.full((n, 10), -4.0, np.float32)
        cls[:, LP.TABLE] = 4.0
        same = np.full((n, n), 5.0, np.float32)
        cells = {i: [([float(v) for v in boxes[i]], t["texts"][i])] for i in range(n)}
        gray = np.full((120, 260), 240, np.uint8)
        (_, _), rec = capture_apply(ocr_v, "ocr", boxes, t["texts"], list(range(n)), cls, same, tab, cells, None, gray, None)
        cv2.imwrite(str(OUT / f"verify-tab-{len(tab_apply)}.png"), gray)
        rec["gray"] = f"tables/verify-tab-{len(tab_apply)}.png"
        tab_apply.append(rec)
    dump(OUT / "verify-crafted.json", {"synthetic": synth, "tab": tabs, "tab_apply": tab_apply})
    return 0


if __name__ == "__main__":
    sys.exit(main())
