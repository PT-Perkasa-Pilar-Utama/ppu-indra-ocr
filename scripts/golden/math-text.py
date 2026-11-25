# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 PT Perkasa Pilar Utama
"""Goldens for the text rules of indra_ocr/math.py and for difflib.SequenceMatcher.

Run by hand from the repo root:

    uvx --from indra-ocr==1.2.0 python -I scripts/golden/math-text.py            # fixtures
    uvx --from indra-ocr==1.2.0 python -I scripts/golden/math-text.py --tables   # py-unicode.ts

Fixtures (CI never runs this script; the output is committed):

- tests/fixtures/golden/math-text.json: every math.accept and math.gate event the
  page goldens recorded (re-run here on the head of its model set, and checked
  against the recorded output), crafted and generated cases for the pure
  functions, and hashes of per-code-point properties over all of Unicode 15.0.
- tests/fixtures/golden/sequence-matcher.json: opcodes, matching blocks and ratio
  of difflib.SequenceMatcher(None, a, b, autojunk=False) on random pairs.

--tables rewrites the generated block of src/indra/py-unicode.ts: the code point
ranges where str.isdigit, unicodedata.combining or a LATIN name apply, and the
GREEK SMALL LETTER names unicodedata.lookup resolves.
"""

from __future__ import annotations

import difflib
import glob
import hashlib
import json
import platform
import random
import re
import sys
import unicodedata as ud
from pathlib import Path

import indra_ocr.math as M

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "fixtures" / "golden"
PY_UNICODE = ROOT / "src" / "indra" / "py-unicode.ts"
MAX = 0x110000
# set name -> model folder (the folder the page goldens ran with)
MODEL_SETS = {"v1.2": ROOT / "models" / "v1.2", "v1.1": ROOT / "models" / "v1.1", "v3.1": ROOT / "models"}


# ---------------------------------------------------------------- tables


def b36(n: int) -> str:
    s = ""
    while True:
        n, r = divmod(n, 36)
        s = "0123456789abcdefghijklmnopqrstuvwxyz"[r] + s
        if not n:
            return s


def table_flags(c: str) -> int:
    return (
        int(ud.combining(c) != 0)
        | int(ud.name(c, "").startswith("LATIN")) << 1
        | int(c.isdigit()) << 2
    )


def flag_table() -> str:
    toks, end, prev = [], -1, None
    runs: list[list[int]] = []
    for cp in range(MAX):
        f = 0 if 0xD800 <= cp < 0xE000 else table_flags(chr(cp))
        if f and prev is not None and prev[2] == f and prev[1] == cp - 1:
            prev[1] = cp
        elif f:
            prev = [cp, cp, f]
            runs.append(prev)
        else:
            prev = None
    for a, b, f in runs:
        toks += [b36(a - end - 1), b36(b - a), b36(f)]
        end = b
    text = " ".join(toks)
    return "\n".join(f'  "{text[i : i + 88]}"' + (" +" if i + 88 < len(text) else "") for i in range(0, len(text), 88))


def greek_table() -> str:
    names = {}
    for cp in range(MAX):
        nm = ud.name(chr(cp), "")
        if nm.startswith("GREEK SMALL LETTER ") and " " not in nm[19:]:
            names[nm[19:].lower()] = cp
    for w, cp in names.items():  # lookup agrees with the names
        assert ord(ud.lookup("GREEK SMALL LETTER " + w.upper())) == cp
    return " ".join(f"{w}:{cp:x}" for w, cp in names.items())


def write_tables() -> None:
    src = PY_UNICODE.read_text(encoding="utf-8")
    block = (
        "// BEGIN GENERATED (scripts/golden/math-text.py --tables)\n"
        f"// Unicode {ud.unidata_version}, Python {platform.python_version()}.\n"
        '/** "gap len flags" triples in base 36: ranges of code points with a table flag. */\n'
        f"const FLAG_TABLE: string =\n{flag_table()};\n"
        '/** "name:code point" pairs of the single-word GREEK SMALL LETTER names. */\n'
        f'const GREEK_TABLE: string =\n  "{greek_table()}";\n'
        "// END GENERATED"
    )
    out = re.sub(r"// BEGIN GENERATED.*?// END GENERATED", lambda _: block, src, flags=re.S)
    PY_UNICODE.write_text(out, encoding="utf-8")
    print("wrote", PY_UNICODE)


# ---------------------------------------------------------------- JSON


def dump(path: Path, obj: dict) -> None:
    """One key per line; one list item per line."""
    parts = []
    for k, v in obj.items():
        if isinstance(v, list):
            body = "[\n" + ",\n".join(json.dumps(x, ensure_ascii=False) for x in v) + "\n]"
        else:
            body = json.dumps(v, ensure_ascii=False)
        parts.append(f"{json.dumps(k)}: {body}")
    path.write_text("{\n" + ",\n".join(parts) + "\n}\n", encoding="utf-8")
    print("wrote", path, f"{path.stat().st_size // 1024} KiB")


# ---------------------------------------------------------------- readers on heads


def load_heads() -> dict:
    """Heads by name: the model sets' own, and two variants that reach the other branches."""
    heads: dict = {"none": {"accept": None, "gate": None}}
    for name, folder in MODEL_SETS.items():
        acc, gate = folder / "math_accept.json", folder / "math_gate.json"
        if acc.exists() or gate.exists():
            heads[name] = {
                "accept": json.loads(acc.read_text(encoding="utf-8")) if acc.exists() else None,
                "gate": json.loads(gate.read_text(encoding="utf-8")) if gate.exists() else None,
            }
    a = heads["v1.2"]["accept"]
    # tau 0: every reading with a $ passes, so the splice runs on the line's own text
    heads["v1.2-tau0"] = {"accept": {**a, "tau": 0.0}, "gate": None}
    # one feature fewer: the head does not know the table flag, so x stays at 7 entries
    heads["v1.2-no-table"] = {
        "accept": {**a, "w": a["w"][:-1], "mu": a["mu"][:-1], "sd": a["sd"][:-1]},
        "gate": None,
    }
    return heads


def reader(head: dict, splice: bool = True) -> M.MathReader:
    r = M.MathReader.__new__(M.MathReader)  # no ONNX sessions: only the head and the flag are read
    r.accept_head, r.gate_head = head["accept"], head["gate"]
    r.prose_splice = splice
    return r


# ---------------------------------------------------------------- recorded events


def recorded_events(heads: dict) -> tuple[list, list]:
    acc, gate = [], []
    for f in sorted(glob.glob(str(GOLDEN / "**" / "*.json"), recursive=True)):
        p = Path(f)
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or "events" not in d:
            continue
        rel = p.relative_to(GOLDEN)
        sset = rel.parts[0] if len(rel.parts) > 1 else "v1.2"
        head = sset if sset in heads else "none"
        for e in d["events"]:
            if e.get("stage") == "math.accept":
                r = reader(heads[head])
                out = r.accept(e["text"], e["conf"], e["draft"], e["rconf"], e["in_table"])
                assert out == e["out"], (rel, e, out)  # the head and the harness are the recorded ones
                acc.append({"set": sset, "case": rel.stem, "head": head, **{k: e[k] for k in ("text", "conf", "draft", "rconf", "in_table", "out")}})
            elif e.get("stage") == "math.gate":
                r = reader(heads[head])
                keep = r.gate(e["probs"], e["drafts"], e["n_cand"], e["source"])
                assert keep == e["keep"], (rel, e, keep)
                gate.append({"set": sset, "case": rel.stem, "head": head, **{k: e[k] for k in ("probs", "drafts", "n_cand", "source", "keep")}})
    return acc, gate


# ---------------------------------------------------------------- crafted cases

# (reading as MathRec writes it, draft as the recognizer or the text layer reads it)
HAND = [
    # accents: Czech, Polish, Vietnamese
    ("Z duvodu $x ^ { 2 }$ je to tak", "Z d\u016fvodu $x^2$ je to tak"),
    ("Pri $x = 1$ plati $y > 0$", "P\u0159i $x=1$ plat\u00ed $y>0$"),
    ("Hodnota $\\sum _ { i } a _ i$ je kladna", "Hodnota \u2211 a_i je kladn\u00e1"),
    ("Pro vsechna $n \\geq 1$ plati", "Pro v\u0161echna $n\\geq 1$ plat\u00ed"),
    ("Dla $x ^ { 2 }$ zolw", "Dla x\u00b2 \u017c\u00f3\u0142w"),
    ("Wartosc $f ( x ) = \\frac { 1 } { 2 }$ jest stala", "Warto\u015b\u0107 $f(x)=\\frac{1}{2}$ jest sta\u0142a"),
    ("Zazolc gesla jazn $a _ 1$", "Za\u017c\u00f3\u0142\u0107 g\u0119\u015bl\u0105 ja\u017a\u0144 $a_1$"),
    ("Viet Nam $a _ { 1 }$ va $b _ { 2 }$", "Vi\u1ec7t Nam $a_1$ v\u00e0 $b_2$"),
    ("Viet Nam $a _ { 1 }$ va $b _ { 2 }$", "Vie\u0302\u0323t Nam $a_1$ va\u0300 $b_2$"),
    ("Nguyen $x ^ 2$ Truong", "Nguy\u1ec5n $x^2$ Tr\u01b0\u1eddng"),
    ("Cac so $n ^ { 2 }$ duoc", "C\u00e1c s\u1ed1 $n^2$ \u0111\u01b0\u1ee3c"),
    # accents written as math
    ("$\\hat { e }$ t $x _ { i }$", "\u00e9t\u00e9 $x_i$"),
    ("caf $\\hat { e }$ au lait $x ^ 2$", "caf\u00e9 au lait $x^2$"),
    ("$\\bar { z }$ ona $\\dot { \\epsilon }$ x", "\u017cona \u00e9x"),
    ("n $\\tilde { a }$ o $\\acute { e }$ $y ^ 2$", "n\u00e3o \u00e9 $y^2$"),
    ("$\\hat { e }$ $\\hat { e }$", "\u00e9\u00e9"),
    ("$\\hat { e }$ $\\hat { e }$", "ee"),
    ("a $\\check { s }$ b $\\ddot { u }$ c $z _ 1$", "a \u0161 b \u00fc c $z_1$"),
    ("a $\\breve { a }$ b", "a \u0103 b"),
    ("$\\grave { \\epsilon }$ $\\mathrm { x } ^ 2$", "\u00e8 x\u00b2"),
    # Cyrillic and Greek prose around formulas
    ("Toni $x _ { i }$ dopiBHioe", "\u0422\u043e\u0434\u0456 $x_i$ \u0434\u043e\u0440\u0456\u0432\u043d\u044e\u0454"),
    ("\u0422\u043e\u0434\u0456 $x _ { i }$ \u0434\u043e\u0440\u0456\u0432\u043d\u044e\u0454", "\u0422\u043e\u0434\u0456 $x_i$ \u0434\u043e\u0440\u0456\u0432\u043d\u044e\u0454"),
    ("\u0414\u043e\u0432\u0435\u0434\u0435\u043c\u043e \u0449\u043e $a ^ 2 + b ^ 2 = c ^ 2$", "\u0414\u043e\u0432\u0435\u0434\u0435\u043c\u043e \u0449\u043e $a^2+b^2=c^2$"),
    ("Pust $f ( x ) = x ^ 2$ togda", "\u041f\u0443\u0441\u0442\u044c $f(x)=x^2$ \u0442\u043e\u0433\u0434\u0430"),
    ("Otan $\\alpha > 0$ tote", "\u038c\u03c4\u03b1\u03bd $\\alpha>0$ \u03c4\u03cc\u03c4\u03b5"),
    ("Ia $\\beta = 2 \\pi$ ypo", "\u0399\u03b1 $\\beta=2\\pi$ \u03c5\u03c0\u03bf"),
    ("\u0414\u043b\u044f $n ^ 2$ \u043c\u0430\u0454\u043c\u043e $\\sum _ { k } a _ k$", "\u0414\u043b\u044f $n^2$ \u043c\u0430\u0454\u043c\u043e $\\sum_k a_k$"),
    ("Let $\\varepsilon > 0$ and $\\vartheta$", "Let $\\varepsilon>0$ and $\\vartheta$"),
    # Azerbaijani schwa, Icelandic thorn, Turkish, others
    ("Burada molumdur $\\sigma _ { 1 }$ ana", "Burada m\u0259lumdur \\sigma_1 ana"),
    ("Burada molumdur $x ^ 2$ ana", "Burada m\u0259lumdur $x^2$ ana"),
    ("Bu molumdur ki $y = 2$ dir", "Bu m\u0259lumdur ki $y=2$ dir"),
    ("Pa er $x ^ 2$ og dad", "\u00de\u00e1 er $x^2$ og \u00fea\u00f0"),
    ("pa er $x ^ 2$ og dad", "\u00fe\u00e1 er $x^2$ og \u00fea\u00f0"),
    ("Hann sagdi $a _ 1$ og", "Hann sag\u00f0i $a_1$ og"),
    ("Dogrusu $f ( x )$ bulunur", "Do\u011frusu $f(x)$ bulunur"),
    ("Ince $x ^ 2$ k\u0131s\u0131m", "\u0130nce $x^2$ k\u0131s\u0131m"),
    ("Strasse $x ^ 2$ Fuss", "Stra\u00dfe $x^2$ Fu\u00df"),
    ("Oeuvre $x ^ 2$ aeon", "\u0152uvre $x^2$ \u00e6on"),
    ("Dzien $x ^ 2$ lodz", "Dzie\u0144 $x^2$ \u0142\u00f3d\u017a"),
    ("Hlavni $x ^ 2$ strojove", "Hlavn\u00ed $x^2$ strojov\u00e9"),
    ("Sorensen $x ^ 2$ Oslo", "S\u00f8rensen $x^2$ \u00d8lso"),
    # odd and unusual numbers of $
    ("a $x ^ 2 b", "a x\u00b2 b"),
    ("a $x ^ 2$ b $y", "a x\u00b2 b y"),
    ("$$x ^ 2$$", "x\u00b2"),
    ("a $ b $ c $ d", "a b c d"),
    ("$", ""),
    ("$$", "x"),
    ("", ""),
    ("", "abc"),
    ("abc", ""),
    ("no math here", "no math here"),
    ("Burada \\sigma ana $x ^ 2$", "Burada \\sigma ana $x^2$"),
    ("Burada \\sigma ana", "Burada \u03c3 ana"),
    ("$x ^ 2$ \\alpha text", "x\u00b2 \u03b1 text"),
    ("price $5 and $6", "price $5 and $6"),
    ("$x ^ 2$$y ^ 2$", "x\u00b2y\u00b2"),
    ("tail $x ^ 2$", "tail x\u00b2"),
    ("$x ^ 2$ head", "x\u00b2 head"),
    # all-math lines
    ("$x ^ { 2 } + y ^ { 2 }$", "x\u00b2+y\u00b2"),
    ("$\\frac { a } { b }$", "a/b"),
    ("$\\sum _ { i = 1 } ^ { n } i$", "\u2211 i"),
    ("$\\int _ 0 ^ 1 f ( x ) d x$", "\u222b f(x)dx"),
    ("$a$", "a"),
    ("$12$", "12"),
    # style commands and spacing
    ("$\\mathrm { dx }$", "dx"),
    ("see $\\mathrm { dx }$ here", "see dx here"),
    ("see $\\mathbf { v } ^ 2$ here", "see v\u00b2 here"),
    ("$\\text { if } x ^ 2$", "if x\u00b2"),
    ("see $\\text { if }$ here", "see if here"),
    ("$\\, \\quad x$ and $\\qquad y$", "x and y"),
    ("a $\\operatorname { sin } x$ b", "a sin x b"),
    ("a $\\mathrm { \\% } 5$ b", "a % 5 b"),
    ("a $\\mathcal { L } _ 1$ b", "a L1 b"),
    ("a $\\mathrm\u00e9 x$ b", "a x b"),
    ("a $\\mathrm2 x$ b", "a x b"),
    ("a $\\mathrm\u2082 x$ b", "a x b"),
    ("a $\\mathrm\u0663 x$ b", "a x b"),
    ("a $\\mathrm_ x$ b", "a x b"),
    ("a $\\mathrm x$ b", "a x b"),
    ("a $\\mathrm{x}$ b", "a x b"),
    ("a $\\text\u00e9 x$ b", "a x b"),
    ("a $\\quad\u00e9 x$ b", "a x b"),
    ("a $\\quadx$ b", "a x b"),
    ("a $\\qquad x$ b", "a x b"),
    ("a $\\textbf { x }$ b", "a x b"),
    ("a $\\textbf { x } _ 1$ b", "a x1 b"),
    ("a $\\text { x } ^ { \\mathrm { y } }$ b", "a x^y b"),
    ("a $\\mathrm { x }$ b $\\mathbf { y }$ c", "a x b y c"),
    # symbol-only spans: plain text has these characters
    ("$66 \\pm 5$ kg", "66 \u00b1 5 kg"),
    ("$\\leq$", "\u2264"),
    ("a $\\leq$ b $\\geq$ c", "a \u2264 b \u2265 c"),
    ("a $\\rightarrow$ b", "a \u2192 b"),
    ("a $\\to$ b $\\times$ c $\\cdot$ d", "a \u2192 b \u00d7 c \u00b7 d"),
    ("a $\\alpha$ b", "a \u03b1 b"),
    ("a $\\pmx$ b", "a \u00b1 b"),
    ("a $\\pm x$ b", "a \u00b1 x b"),
    ("a $\\degree C$ b", "a \u00b0 C b"),
    ("a $5 \\circ$ b", "a 5\u00b0 b"),
    ("a $\\ldots$ b $\\cdots$ c", "a ... b ... c"),
    ("a $\\prime$ b", "a \u2032 b"),
    ("a $x \\mid y$ b", "a x | y b"),
    ("a $\\Rightarrow$ b $\\Leftrightarrow$ c", "a \u21d2 b \u21d4 c"),
    ("a $\\vert x \\vert$ b", "a |x| b"),
    ("a $x \\neq y$ b $x \\ne y$", "a x \u2260 y b x \u2260 y"),
    ("a $x \\approx y$ b", "a x \u2248 y b"),
    ("a $x \\sim y$ b $\\simeq$ c", "a x ~ y b \u2243 c"),
    ("a $x _ 1$ b", "a x\u2081 b"),
    ("a $x ^ 2$ b", "a x\u00b2 b"),
    # Python whitespace inside spans and around braces
    ("a $\\mathrm\u00a0{ x } ^ 2$ b", "a x\u00b2 b"),
    ("a $\\mathrm\u0085{x} ^ 2$ b", "a x\u00b2 b"),
    ("a $\\textbf\u3000{ x } _ 1$ b", "a x1 b"),
    ("a $x\ufeff^ 2$ b", "a x2 b"),
    ("a $\\hat\u3000{\u00a0e\u3000}$ b", "a \u00e9 b"),
    ("a $\\hat\u001c{ e }$ b", "a \u00e9 b"),
    ("a $\\hat\ufeff{ e }$ b $x ^ 2$", "a \u00e9 b $x^2$"),
    ("a $\\frac\u00a0{ 1 } { 2 }$ b", "a 1/2 b"),
    ("a $\\frac\ufeff{ 1 } { 2 }$ b", "a 1/2 b"),
    ("a $\\sqrt\u0085{ 2 }$ b", "a \u221a2 b"),
    # LaTeX outside the spans
    ("a \\alpha b $x ^ 2$", "a \u03b1 b x\u00b2"),
    ("a $x ^ 2$ \\frac b", "a x\u00b2 b"),
    ("a $x ^ 2$ b ^ c", "a x\u00b2 b ^ c"),
    ("a $x ^ 2$ b _ c", "a x\u00b2 b _ c"),
    ("a $x ^ 2$ b { c }", "a x\u00b2 b { c }"),
    ("a $x ^ 2$ \\ b", "a x\u00b2 b"),
    ("a $x ^ 2$ \\\u00e9 b", "a x\u00b2 b"),
    # code points outside the BMP, odd whitespace
    ("a \U0001f600 $x ^ 2$ b", "a \U0001f600 $x^2$ b"),
    ("a \U0001d465 $x ^ 2$ b", "a \U0001d465 $x^2$ b"),
    ("\U0001d465 is $x ^ 2$", "\U0001d465 is $x^2$"),
    ("a\u00a0$x ^ 2$\u00a0b", "a\u00a0$x^2$\u00a0b"),
    ("a\u3000$x ^ 2$\u3000b", "a $x^2$ b"),
    ("a\u0085$x ^ 2$\u001cb", "a\u0085$x^2$\u001cb"),
    ("a\ufeff$x ^ 2$ b", "a\ufeff$x^2$ b"),
    ("  a $x ^ 2$ b  ", "a $x^2$ b"),
    ("\t a $x ^ 2$ b \n", "a $x^2$ b"),
    ("a $x ^ 2$ b", "   "),
    ("a $x ^ 2$ b", "\u00a0"),
    ("a $x ^ 2$ b", "12 34"),
    ("a $x ^ 2$ b", "!!!"),
    ("fi $x ^ 2$ ffl", "\ufb01 $x^2$ \ufb04"),
    ("dz $x ^ 2$", "\u01c6 $x^2$"),
    ("I $x ^ 2$ i", "\u0130 $x^2$ \u0131"),
    ("Sigma $x ^ 2$ final", "\u03a3\u0391\u03a3 $x^2$ final"),
    ("ABC $x ^ 2$ abc", "\uff21\uff22\uff23 $x^2$ \uff41\uff42\uff43"),
    ("Ka $x ^ 2$ la", "K\u00e5 $x^2$ \u0142a"),
    ("e $x ^ 2$ e", "\u1eb9\u0300 $x^2$ e\u0323\u0300"),
    # Indic and other scripts with combining marks
    ("namaste $x ^ 2$", "\u0928\u092e\u0938\u094d\u0924\u0947 $x^2$"),
    ("Hindi $x ^ 2$ text", "\u0939\u093f\u0928\u094d\u0926\u0940 $x^2$ text"),
    ("Thai $x ^ 2$", "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 $x^2$"),
    ("Arabic $x ^ 2$ text", "\u0645\u0631\u062d\u0628\u0627 $x^2$ text"),
    ("Hebrew $x ^ 2$", "\u05e9\u05dc\u05d5\u05dd $x^2$"),
    ("Chinese $x ^ 2$", "\u4f60\u597d $x^2$"),
    ("Korean $x ^ 2$", "\ud55c\uad6d\uc5b4 $x^2$"),
]

LATIN_LINES = [
    "",
    "123 + 456",
    "!!!",
    "hello world",
    "Hello \u043c\u0438\u0440",
    "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440",
    "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440 hello",
    "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440 hi",
    "\u041f\u0440\u0438\u0432\u0435\u0442 hello world foo bar",
    "\u03b1 + \u03b2 = \u03b3",
    "\u03b1\u03b2 + \u03b3\u03b4",
    "\u038c\u03c4\u03b1\u03bd x = 1",
    "\u038c\u03c4\u03b1\u03bd",
    "x \u03b1 y z w",
    "x \u03b1\u03b2 y z",
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\u0928\u092e\u0938\u094d\u0924\u0947 hello world",
    "\u0939\u093f\u0928\u094d\u0926\u0940",
    "\u0939\u093f\u0928\u094d\u0926\u0940 hello",
    "\u0939\u093f\u0928\u094d\u0926\u0940 \u0939\u093f\u0928\u094d\u0926\u0940 hello world",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a ok",
    "\u0645\u0631\u062d\u0628\u0627",
    "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd",
    "\u4f60\u597d",
    "\u4f60\u597d hello",
    "\u4f60\u597d\u4f60\u597d hello",
    "\ud55c\uad6d\uc5b4 hello",
    "ab\u0301c",
    "\u0301\u0301",
    "a\u0301\u0301 \u041f\u0440\u0438\u0432\u0435\u0442",
    "\u041f\u0440\u0438\u0432\u0435\u0442\u0301 \u043c\u0438\u0440",
    "\u0e01\u0e34\u0e38 abc",
    "\u01c6 \u01c8 \u01cb",
    "\uff21\uff22\uff23",
    "\uff41\uff42\uff43 \uff21\uff22\uff23",
    "\u1d00\u0299\u1d04",
    "\u00aa\u00ba",
    "\u02b0\u02b2",
    "\u2170 \u2171 \u2172",
    "\u03a3",
    "\u03a3\u0391\u03a3 x",
    "\u03a3\u0391\u03a3 xy",
    "\U00010400\U00010401 hello",
    "\U0001d465\U0001d466 hello",
    "\u0133 \u01c6 \ufb01",
    "\u01c5",
    "\ua78c\ua78e",
    "\u2c61 \u2c65",
    "\u1d43\u1d47",
    "\u0250\u0251\u0252 \u0253\u0254",
    "\u02bbokina \u02bc",
    "ka\u0301\u0301 \u0436",
    "\u0436",
    "\u0436\u0437",
    "\u0436\u0437 x",
    "\u0436\u0437 xx",
    "\u0436\u0437\u0438 xx",
    "\u0436\u0437\u0438 xxx",
    "\u0436\u0437\u0438 xxxx",
]

# draft texts for gate_features
GATE_DRAFTS = [
    "",
    "1",
    "x = 2",
    "a_i = a_{0} e^{-\\lambda i}",
    "\u2211 a_i r(t - i) = Q(t)",
    "We write Q(t) for the flow on day t and r(t) for the rain.",
    "Price $5 \u20ac 6 \u00a3 7 \u00a5 8 % 9",
    "ABC DEF ghi",
    "a   b\tc\n d",
    "\u00a0 a \u3000 b",
    "\U0001d465 = \U0001d466",
    "x\u00b2 + y\u2082 = \u2462",
    "\u0661\u0662\u0663 456 \u096a\u096b\u096c",
    "\u03b1\u03b2\u03b3 \u03b4 \u03b5",
    "\u0370\u0371\u0400\u03ff\u036f",
    "\u2202f/\u2202x \u2248 \u221e \u2260 0 \u2264 1 \u2265 2 \u00b1 3 \u00d7 4 \u00f7 5 \u2208 \u2200 \u2203 \u2192 \u2190 |",
    "\u221a2 \u222b f",
    "\u00c0\u00c9\u00ce\u00d5\u00dc \u00e0\u00e9\u00ee\u00f5\u00fc",
    "\u01c5 \u01c8 \u01cb",
    "= = = =",
    "Z duvodu je to tak",
    "Z d\u016fvodu je to tak",
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\u4f60\u597d\uff0c\u4e16\u754c hello",
    "a\u0301\u0301b",
    "  leading and trailing  ",
    "\U0001f600 emoji \U0001f600",
    "\u0391\u0392\u0393",
    "\u01c5",
    "\u2163 \u2173",
    "x" * 300,
    "1" * 40,
]
GATE_ARGS = [  # prob, n_flagged, pmax, n_cand, source
    (0.995511, 5, 1.0, 7, "ocr"),
    (0.5, 1, 0.5, 21, "text"),
    (0.0, 0, 0.0, 0, "ocr"),
    (1.0, 1, 1.0, 1, "other"),
    (0.853804, 12, 0.999993, 100, "text"),
]


CHAR_COLUMNS = ["alpha", "upper", "digit", "space", "lower", "fold", "look", "combining", "cat_m", "latin_name", "nfkd", "nfkc"]


def char_flags(c: str) -> dict:
    return {
        "alpha": c.isalpha(), "upper": c.isupper(), "digit": c.isdigit(), "space": c.isspace(),
        "lower": c.lower(), "fold": M._fold(c), "look": M._look(c), "combining": ud.combining(c) != 0,
        "cat_m": ud.category(c).startswith("M"), "latin_name": ud.name(c, "").startswith("LATIN"),
        "nfkd": ud.normalize("NFKD", c), "nfkc": ud.normalize("NFKC", c),
    }


def assigned_ranges() -> list:
    """Ranges of the code points Unicode 15.0 assigns (category other than Cn), surrogates left out."""
    out: list = []
    for cp in range(MAX):
        if 0xD800 <= cp < 0xE000 or ud.category(chr(cp)) == "Cn":
            continue
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return out


def char_hashes() -> dict:
    """sha256 per property of 'cp:value' lines over every code point Unicode 15.0 assigns."""
    props = ["alpha", "upper", "digit", "space", "alnum", "lower", "fold", "look", "combining", "cat_m", "latin_name", "nfkd", "nfkc"]
    h = {p: hashlib.sha256() for p in props}
    n = 0
    for cp in range(MAX):
        c = chr(cp)
        if 0xD800 <= cp < 0xE000 or ud.category(c) == "Cn":
            continue
        n += 1
        f = char_flags(c)
        f["alnum"] = c.isalnum()
        for p in props:
            v = f[p]
            v = int(v) if isinstance(v, bool) else ".".join(f"{ord(x):x}" for x in v)
            h[p].update(f"{cp:x}:{v}\n".encode())
    return {"assigned": n, **{p: x.hexdigest() for p, x in h.items()}}


def gen_pairs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    """Prose in several scripts around formulas: the draft has the true letters and a plain
    formula, the reading has MathRec's Latin letters and LaTeX spans."""
    words = [
        "d\u016fvodu", "\u017eena", "p\u0159\u00edklad", "\u0142\u00f3d\u017a", "\u017c\u00f3\u0142\u0107", "g\u0119\u015bl\u0105", "vi\u1ec7t", "\u0111\u01b0\u1ee3c", "\u1ec7", "Tr\u01b0\u1eddng",
        "\u0422\u043e\u0434\u0456", "\u0434\u043e\u0440\u0456\u0432\u043d\u044e\u0454", "\u043f\u0440\u0438\u0432\u0435\u0442", "\u043c\u0438\u0440", "\u038c\u03c4\u03b1\u03bd", "\u03c4\u03cc\u03c4\u03b5", "\u03ac", "m\u0259lumdur", "\u0259", "\u00fe\u00e1", "\u00fea\u00f0",
        "Do\u011frusu", "\u0130nce", "hello", "the", "flow", "of", "a", "I", "\u01c6", "\ufb01nal", "Stra\u00dfe",
        "\u00c0\u00c9\u00ce", "na\u00efve", "caf\u00e9", "se\u00f1or", "\u00f1", "\u00f8", "\u00c5", "\u0153uvre", "\u00e7",
    ]
    spans = [
        ("x ^ { 2 }", "x^2"), ("x _ { i }", "x_i"), ("\\frac { a } { b }", "a/b"), ("\\alpha > 0", "\u03b1>0"),
        ("\\sum _ { k } a _ k", "\u2211 a_k"), ("f ( x ) = 1", "f(x)=1"), ("\\mathrm { dx }", "dx"),
        ("n \\geq 1", "n\u22651"), ("\\lambda", "\\lambda"), ("\\varepsilon", "\u03b5"), ("E = m c ^ 2", "E=mc2"),
        ("\\hat { e }", "\u00e9"), ("\\bar { z }", "\u017c"), ("\\pm 5", "\u00b15"), ("66", "66"), ("a", "a"),
    ]
    out = []
    for _ in range(n):
        segs = []
        for _ in range(rng.randint(1, 3)):
            w = " ".join(rng.choice(words) for _ in range(rng.randint(1, 3)))
            s = rng.choice(spans)
            segs.append((w, s))
        reading = " ".join(f"{_asciify(w, rng)} ${s[0]}$" for w, s in segs)
        draft = " ".join(f"{w} ${s[1]}$" if rng.random() < 0.5 else f"{w} {s[1]}" for w, s in segs)
        if rng.random() < 0.4:
            tail = rng.choice(words)
            reading += " " + _asciify(tail, rng)
            draft += " " + tail
        if rng.random() < 0.3 and len(draft) > 4:  # a misread character in the draft
            i = rng.randrange(len(draft))
            draft = draft[:i] + rng.choice("1l|oO0 ") + draft[i + 1 :]
        if rng.random() < 0.15:
            reading = reading.replace("$", "", 1)
        if rng.random() < 0.1:
            draft = ud.normalize("NFD", draft)
        out.append((reading, draft))
    return out


def _asciify(w: str, rng: random.Random) -> str:
    """What MathRec writes for a word: Latin letters without accents, a look-alike for others."""
    return "".join(M._look(c) if rng.random() < 0.85 else rng.choice("oaei") for c in w)


def craft_case(reading: str, draft: str, heads: dict) -> dict:
    p1, p2 = (0.95, 0.93, 0.0), (0.85, 0.55, 1.0)
    acc = {}
    for name in ("none", "v1.2", "v1.2-tau0", "v1.2-no-table"):
        r = reader(heads[name])
        acc[name] = [r.accept(reading, p[0], draft, p[1], p[2]) for p in (p1, p2)]
    acc["none-no-splice"] = [reader(heads["none"], splice=False).accept(reading, p1[0], draft, p1[1], p1[2])]
    acc["v1.2-tau0-no-splice"] = [reader(heads["v1.2-tau0"], splice=False).accept(reading, p1[0], draft, p1[1], p1[2])]
    return {
        "reading": reading,
        "draft": draft,
        "has_math": M.has_math(reading),
        "has_structure": M.has_structure(reading),
        "accept_math": M.accept_math(reading, draft),
        "restore_accents": M.restore_accents(reading, draft),
        "splice_prose": M.splice_prose(reading, draft),
        "units": [[u, a, b] for u, a, b in M._units(reading)],
        "units_draft": [[u, a, b] for u, a, b in M._units(draft)],
        "latin_line": M.latin_line(reading),
        "sig_reading": M._sig_reading(reading),
        "sig_draft": M._sig_draft(draft),
        "accept": acc,
    }


def text_like_runs() -> list:
    seqs = [
        ["$", "x", "$"], ["\\alpha", "$", "x", "$"], ["a", "\\frac", "b"], ["$", "\\frac", "$", "\\sum"],
        ["\\", "x"], ["\\1", "\\ "], ["$", "$", "\\alpha"], ["\\\u00e9", "\\\u03b1"], ["\\\u00e9"], ["$", "\\\u00e9", "$", "\\\u00e9"],
        ["ab", "\\a", "$", "\\a", "$", "$", "\\a"], ["\\\u0663"], ["\\\u00b2"], [""], ["\\"],
    ]
    out = []
    for s in seqs:
        flags, in_math = [], [False]
        for t in s:
            flags.append(M.MathReader._text_like(0, t, in_math))
        out.append({"tokens": s, "flags": flags, "in_math": in_math[0]})
    return out


def math_fixture() -> dict:
    heads = load_heads()
    acc, gate = recorded_events(heads)
    rng = random.Random(20261125)
    pairs = HAND + gen_pairs(rng, 200)
    crafted = [craft_case(r, d, heads) for r, d in pairs]
    tokens = json.loads((MODEL_SETS["v1.2"] / "math_meta.json").read_text(encoding="utf-8"))["tokens"]
    chars = frozenset(t for t in tokens if len(t) == 1)
    wr = reader(heads["none"])
    wr.chars = chars
    latin_chars = frozenset(c for c in chars if c.isascii())
    gate_features = []
    for d in GATE_DRAFTS:
        for args in GATE_ARGS:
            gate_features.append({"draft": d, "args": list(args), "x": M.gate_features(args[0], d, *args[1:])})
    gate_cases = []
    for probs, drafts, n_cand, source in [
        ([0.9, 0.5, 0.99], ["x = 2", "", "We write Q(t) for the flow"], 7, "ocr"),
        ([0.5], ["1"], 21, "ocr"),
        ([], [], 3, "text"),
        ([1.0, 0.7], ["\u2211 a_i = 1", "Hello"], 2, "text"),
        ([0.6, 0.6, 0.6], ["a", "b", "c"], 40, "ocr"),
    ]:
        gate_cases.append({
            "probs": probs, "drafts": drafts, "n_cand": n_cand, "source": source,
            "keep": {n: reader(heads[n]).gate(probs, drafts, n_cand, source) for n in ("none", "v1.2")},
        })
    writes = []
    for t in [d for _, d in pairs[:60]] + LATIN_LINES:
        for share in (0.9, 0.5, 1.0):
            writes.append({"text": t, "share": share, "full": wr.writes(t, share), "latin": _writes(latin_chars, t, share)})
    sample = "".join(chr(c) for r in [(0x20, 0x17F), (0x180, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x45F), (0x1E00, 0x1EFF), (0x2070, 0x209F), (0x2460, 0x2470), (0xFB00, 0xFB06), (0xFF21, 0xFF3A), (0x1D400, 0x1D41F), (0x900, 0x97F), (0xE00, 0xE3F)] for c in range(*r))
    return {
        "indra_ocr": "1.2.0",
        "python": platform.python_version(),
        "unicode": ud.unidata_version,
        "constants": {"MATH_MIN_CONF": M.MATH_MIN_CONF, "SPLICE_MIN_SEG": M.SPLICE_MIN_SEG, "SPLICE_MIN_LINE": M.SPLICE_MIN_LINE},
        "heads": heads,
        "tokens": "".join(sorted(chars)),
        "recorded_accept": acc,
        "recorded_gate": gate,
        "crafted": crafted,
        "latin_lines": [{"text": t, "latin": M.latin_line(t)} for t in LATIN_LINES],
        "gate_features": gate_features,
        "gate_cases": gate_cases,
        "writes": writes,
        "text_like": text_like_runs(),
        "char_columns": CHAR_COLUMNS,
        "chars": [[c, *[(int(v) if isinstance(v, bool) else v) for v in (char_flags(c)[k] for k in CHAR_COLUMNS)]] for c in sample],
        "assigned_ranges": assigned_ranges(),
        "char_hashes": char_hashes(),
        "greek": {w: _greek(w) for w in GREEK_WORDS},
    }


GREEK_WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "lamda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "heta", "sampi", "digamma", "koppa", "stigma", "san", "sho", "yot", "sum", "ell", "i", "var", "iable", "final", "nabla", "partial"]


def _greek(w: str) -> str | None:
    try:
        return ud.lookup("GREEK SMALL LETTER " + w.upper())
    except KeyError:
        return None


def _writes(chars: frozenset, text: str, share: float) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c in chars for c in letters) >= share * len(letters)


# ---------------------------------------------------------------- SequenceMatcher


def matcher_fixture() -> dict:
    rng = random.Random(1202026)
    cases = []

    def add(a, b, flm: int = 0) -> None:
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        case = {
            "a": a, "b": b,
            "opcodes": [list(o) for o in sm.get_opcodes()],
            "blocks": [[m.a, m.b, m.size] for m in sm.get_matching_blocks()],
            "ratio": sm.ratio(),
        }
        if flm:
            case["flm"] = []
            for _ in range(flm):
                alo = rng.randint(0, len(a))
                blo = rng.randint(0, len(b))
                ahi, bhi = rng.randint(alo, len(a)), rng.randint(blo, len(b))
                m = sm.find_longest_match(alo, ahi, blo, bhi)
                case["flm"].append([alo, ahi, blo, bhi, m.a, m.b, m.size])
        cases.append(case)

    def rstr(alpha: str, n: int) -> str:
        return "".join(rng.choice(alpha) for _ in range(n))

    def mutate(seq, alphabet):
        seq = list(seq)
        for _ in range(rng.randint(0, 6)):
            op = rng.choice("dis")
            i = rng.randrange(len(seq) + 1)
            if op == "d" and seq:
                del seq[min(i, len(seq) - 1)]
            elif op == "i":
                seq.insert(i, rng.choice(alphabet))
            elif seq:
                seq[min(i, len(seq) - 1)] = rng.choice(alphabet)
        return seq

    for alpha in ("ab", "abc", "abcd", "ab ", "abcdefgh"):  # few letters: many ties
        for _ in range(40):
            a = rstr(alpha, rng.randint(0, 30))
            b = rstr(alpha, rng.randint(0, 30))
            add(a, b, flm=3)
    for _ in range(40):  # b is a damaged copy of a
        a = rstr("abc", rng.randint(0, 60))
        add(a, "".join(mutate(a, "abcx")), flm=2)
    for _ in range(40):  # lists of strings, with empty and multi-character elements
        alpha = ["a", "b", "c", "ab", "", "x y", "\u00e9"]
        a = [rng.choice(alpha) for _ in range(rng.randint(0, 25))]
        b = [rng.choice(alpha) for _ in range(rng.randint(0, 25))]
        add(a, b, flm=3)
    for _ in range(20):  # folded letters, as restore_accents and splice_prose call it
        alpha = ["a", "e", "i", "o", "u", "z", "\0", "x"]
        a = [rng.choice(alpha) for _ in range(rng.randint(5, 40))]
        add(a, mutate(a, alpha), flm=2)
    for _ in range(20):  # code points outside the BMP and combining marks
        alpha = "a\U0001f600\U0001d465\u0301\u00e9"
        add(rstr(alpha, rng.randint(0, 20)), rstr(alpha, rng.randint(0, 20)), flm=2)
    for _ in range(10):  # 250 to 400 elements: popular elements stay in b2j with autojunk=False
        n = rng.randint(250, 400)
        a = rstr("abc", n)
        add(a, "".join(mutate(a, "abc")) if rng.random() < 0.5 else rstr("abc", rng.randint(200, 400)))
    add("", "")
    add("abc", "abc")
    add("abc", "")
    add("", "abc")
    add(" abcd", "abcd abcd", flm=4)
    add("qabxcd", "abycdf")
    add("ab", "acab")
    add("abxcd", "abcd")
    return {"python": platform.python_version(), "cases": cases}


def main() -> None:
    if "--tables" in sys.argv:
        write_tables()
        return
    dump(GOLDEN / "sequence-matcher.json", matcher_fixture())
    dump(GOLDEN / "math-text.json", math_fixture())


if __name__ == "__main__":
    main()
