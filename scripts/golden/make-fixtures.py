"""Generate the parity fixtures in tests/fixtures/pages/.

Run by hand from the repo root:

    DYLD_LIBRARY_PATH=/opt/homebrew/lib \
        uvx --with pillow --with reportlab --with matplotlib python -I scripts/golden/make-fixtures.py

Pillow needs fribidi (brew install fribidi) for the Hebrew, Thai and Devanagari
lines; DYLD_LIBRARY_PATH lets its wheel find the Homebrew library.

All text is written for this project. Fonts come from macOS system folders;
the fixtures hold rendered glyphs, not font files. The output is committed, so
CI never runs this script.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "fixtures" / "pages"

DPI = 200
W, H = int(8.5 * DPI), int(11 * DPI)  # US letter at 200 dpi, upstream's default render dpi
MARGIN = 160

FONT_DIR = Path("/System/Library/Fonts")
SERIF = str(FONT_DIR / "Supplemental" / "Times New Roman.ttf")
SERIF_BOLD = str(FONT_DIR / "Supplemental" / "Times New Roman Bold.ttf")
SANS = str(FONT_DIR / "Helvetica.ttc")
CJK = str(FONT_DIR / "STHeiti Light.ttc")
SUPP = FONT_DIR / "Supplemental"
KOREAN = str(FONT_DIR / "AppleSDGothicNeo.ttc")
THAI = str(SUPP / "Thonburi.ttc")
HINDI = str(FONT_DIR / "Kohinoor.ttc")
HEBREW = str(FONT_DIR / "ArialHB.ttc")
UNICODE_TTF = str(SUPP / "Arial Unicode.ttf")  # TrueType CJK and Hebrew glyphs for reportlab


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def paragraph(draw, x, y, text, f, width, leading=1.35) -> int:
    for line in wrap(draw, text, f, width):
        draw.text((x, y), line, font=f, fill="black")
        y += int(f.size * leading)
    return y + int(f.size * 0.6)


def page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("L", (W, H), 255)
    return img, ImageDraw.Draw(img)


def furniture(draw, title: str, number: int) -> None:
    small = font(SANS, 26)
    draw.text((MARGIN, 80), title, font=small, fill="black")
    draw.text((W - MARGIN - 40, H - 110), str(number), font=small, fill="black")


COLUMN_TEXT = [
    (
        "1. Field notes",
        "The survey team walked the eastern ridge in the first week of May. "
        "They recorded each spring, its flow in litres per minute, and the "
        "colour of the water. Most springs ran clear. Two ran brown after rain "
        "and cleared again within a day.",
    ),
    (
        "2. Equipment",
        "Each team carried a hand pump, a set of sample bottles, a notebook "
        "and a small scale. The scale drifted by a few grams over the week, so "
        "the team checked it against a reference weight every morning.",
    ),
    (
        "3. Results",
        "The ridge holds forty-one springs. Thirty-six of them flow all year. "
        "The five seasonal springs sit on the south slope, where the soil is "
        "thin and the rock is close to the surface.",
    ),
    (
        "4. Next steps",
        "The team will return in October to measure the same springs at the "
        "end of the dry season. They will also map the two brown springs and "
        "look for the source of the silt.",
    ),
]


def two_column() -> Image.Image:
    img, d = page()
    furniture(d, "Ridge Water Survey - Spring Report", 3)
    y = 200
    d.text((MARGIN, y), "Springs of the Eastern Ridge", font=font(SERIF_BOLD, 64), fill="black")
    y += 120
    body = font(SERIF, 34)
    head = font(SERIF_BOLD, 40)
    gutter = 80
    col_w = (W - 2 * MARGIN - gutter) // 2
    xs = [MARGIN, MARGIN + col_w + gutter]
    for col, chunk in enumerate([COLUMN_TEXT[:2], COLUMN_TEXT[2:]]):
        cy = y
        for title, text in chunk:
            d.text((xs[col], cy), title, font=head, fill="black")
            cy += 70
            cy = paragraph(d, xs[col], cy, text, body, col_w)
            cy = paragraph(d, xs[col], cy, text, body, col_w)
    return img


def table() -> Image.Image:
    img, d = page()
    furniture(d, "Ridge Water Survey - Appendix", 7)
    y = 220
    d.text((MARGIN, y), "Appendix A. Spring flow", font=font(SERIF_BOLD, 56), fill="black")
    y += 110
    body = font(SERIF, 34)
    y = paragraph(
        d,
        MARGIN,
        y,
        "The table lists the five largest springs. Flow is the mean of three "
        "readings taken on different days.",
        body,
        W - 2 * MARGIN,
    )
    y += 20
    rows = [
        ["Spring", "Slope", "Flow (L/min)", "Colour"],
        ["North Gate", "North", "42.5", "Clear"],
        ["Fern Hollow", "East", "37.0", "Clear"],
        ["Red Stone", "South", "21.8", "Brown"],
        ["Old Mill", "West", "19.4", "Clear"],
        ["Cedar Step", "East", "15.1", "Clear"],
    ]
    widths = [430, 260, 360, 300]
    row_h = 74
    x0 = MARGIN
    for r, row in enumerate(rows):
        x = x0
        f = font(SERIF_BOLD if r == 0 else SERIF, 34)
        for c, cell in enumerate(row):
            d.text((x + 20, y + 18), cell, font=f, fill="black")
            x += widths[c]
        y += row_h
    top = y - row_h * len(rows)
    right = x0 + sum(widths)
    for r in range(len(rows) + 1):
        d.line([(x0, top + r * row_h), (right, top + r * row_h)], fill="black", width=3)
    x = x0
    for c in range(len(widths) + 1):
        d.line([(x, top), (x, y)], fill="black", width=3)
        if c < len(widths):
            x += widths[c]
    y += 30
    d.text((MARGIN, y), "Table 1. Mean flow of the five largest springs.", font=font(SERIF, 30), fill="black")
    return img


CJK_LINES = [
    "山泉调查报告",
    "调查队在五月第一周走访了东岭。",
    "他们记录了每一处泉水的流量和水的颜色。",
    "大多数泉水清澈，雨后有两处变成褐色。",
    "第二天这两处泉水又恢复清澈。",
    "东岭共有四十一处泉水，其中三十六处常年有水。",
    "调查队将在十月再次测量这些泉水。",
    "Report 2026-05, East Ridge, 41 springs.",
]


def chinese() -> Image.Image:
    img, d = page()
    y = 220
    d.text((MARGIN, y), CJK_LINES[0], font=font(CJK, 64), fill="black")
    y += 140
    f = font(CJK, 40)
    for line in CJK_LINES[1:]:
        d.text((MARGIN, y), line, font=f, fill="black")
        y += 80
    return img


def mathtext(text: str, size: int) -> Image.Image:
    """A line of text with inline $...$ math, rendered by matplotlib mathtext."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, text, fontsize=size)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=DPI, format="png", bbox_inches="tight", pad_inches=0.02, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("L")


MATH_LINES = [
    r"We write $Q(t)$ for the flow on day $t$ and $r(t)$ for the rain.",
    r"The flow is a weighted sum $Q(t) = \sum_{i=0}^{n} a_i r(t-i)$ over past days.",
    r"Each weight decays as $a_i = a_0 e^{-\lambda i}$ with $\lambda > 0$.",
    r"The half life is $t_{1/2} = \ln 2 / \lambda$, about four days.",
    r"So $a_7 \approx a_0 / 3$ when $\lambda = 0.17$.",
]


def math_page() -> Image.Image:
    img, d = page()
    y = 220
    d.text((MARGIN, y), "Flow model", font=font(SERIF_BOLD, 56), fill="black")
    y += 120
    for line in MATH_LINES:
        im = mathtext(line, 12)
        img.paste(im, (MARGIN, y))
        y += im.height + 36
    return img


def digital_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    pw, ph = letter
    m = 72
    c.setFont("Helvetica", 9)
    c.drawString(m, ph - 40, "Ridge Water Survey - Digital Report")
    c.setFont("Times-Bold", 22)
    c.drawString(m, ph - 90, "Springs of the Eastern Ridge")
    gutter = 28
    col_w = (pw - 2 * m - gutter) / 2
    for col, chunk in enumerate([COLUMN_TEXT[:2], COLUMN_TEXT[2:]]):
        x = m + col * (col_w + gutter)
        y = ph - 130
        for title, text in chunk:
            c.setFont("Times-Bold", 14)
            c.drawString(x, y, title)
            y -= 20
            c.setFont("Times-Roman", 11)
            words, line = text.split(), ""
            for word in words + [""]:
                trial = f"{line} {word}".strip()
                if word and c.stringWidth(trial, "Times-Roman", 11) <= col_w:
                    line = trial
                    continue
                c.drawString(x, y, line)
                y -= 14
                line = word
            y -= 10
    c.setFont("Helvetica", 9)
    c.drawString(pw - m, 40, "1")
    c.showPage()
    c.save()


def locked_pdf(path: Path) -> None:
    """A one-line PDF encrypted with the user password "secret"."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter, invariant=1, encrypt="secret")
    c.setFont("Times-Roman", 14)
    c.drawString(72, letter[1] - 100, "This page is locked with a password.")
    c.showPage()
    c.save()


def scanned_pdf(path: Path, image: Image.Image) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    c.drawImage(ImageReader(buf), 0, 0, width=letter[0], height=letter[1])
    c.showPage()
    c.save()


MULTISCRIPT_LINES = [
    (KOREAN, "\uc870\uc0ac\ud300\uc740 5\uc6d4 \uccab\uc9f8 \uc8fc\uc5d0 \ub3d9\ucabd \ub2a5\uc120\uc744 \uac78\uc5c8\uc2b5\ub2c8\ub2e4."),
    (KOREAN, "\ub300\ubd80\ubd84\uc758 \uc0d8\ubb3c\uc740 \ub9d1\uc558\uace0, \ub450 \uacf3\uc740 \ube44\uac00 \uc628 \ub4a4 \uac08\uc0c9\uc73c\ub85c \ubcc0\ud588\uc2b5\ub2c8\ub2e4."),
    (SERIF, "Spring survey 2026, East Ridge: 41 springs recorded."),
    (THAI, "\u0e17\u0e35\u0e21\u0e2a\u0e33\u0e23\u0e27\u0e08\u0e40\u0e14\u0e34\u0e19\u0e15\u0e32\u0e21\u0e2a\u0e31\u0e19\u0e40\u0e02\u0e32\u0e14\u0e49\u0e32\u0e19\u0e15\u0e30\u0e27\u0e31\u0e19\u0e2d\u0e2d\u0e01"),
    (THAI, "\u0e19\u0e49\u0e33\u0e1e\u0e38\u0e2a\u0e48\u0e27\u0e19\u0e43\u0e2b\u0e0d\u0e48\u0e43\u0e2a\u0e2a\u0e30\u0e2d\u0e32\u0e14"),
    (HINDI, "\u0938\u0930\u094d\u0935\u0947\u0915\u094d\u0937\u0923 \u0926\u0932 \u0928\u0947 \u092e\u0908 \u0915\u0947 \u092a\u0939\u0932\u0947 \u0938\u092a\u094d\u0924\u093e\u0939 \u092e\u0947\u0902 \u092a\u0942\u0930\u094d\u0935\u0940 \u092a\u0939\u093e\u0921\u093c\u0940 \u0915\u093e \u0926\u094c\u0930\u093e \u0915\u093f\u092f\u093e\u0964"),
]
BANNER = "\ubd04\ucca0 \uc0d8\ubb3c \uc870\uc0ac \ubcf4\uace0\uc11c"  # the Korean title, light on dark


def multiscript() -> Image.Image:
    """Lines for every recognizer group (Korean, Thai, Devanagari, Latin) under
    a light-on-dark title band."""
    img, d = page()
    d.rectangle([MARGIN - 20, 190, W - MARGIN + 20, 310], fill=30)
    d.text((MARGIN, 210), BANNER, font=font(KOREAN, 64), fill=245)
    y = 400
    for path, line in MULTISCRIPT_LINES:
        d.text((MARGIN, y), line, font=font(path, 44), fill="black")
        y += 110
    return img


HEBREW_TITLE = "\u05e1\u05e7\u05e8 \u05d4\u05de\u05e2\u05d9\u05d9\u05e0\u05d5\u05ea \u05d1\u05e8\u05db\u05e1 \u05d4\u05de\u05d6\u05e8\u05d7\u05d9"
HEBREW_TEXT = [
    "\u05e6\u05d5\u05d5\u05ea \u05d4\u05e1\u05e7\u05e8 \u05d4\u05dc\u05da \u05dc\u05d0\u05d5\u05e8\u05da \u05d4\u05e8\u05db\u05e1 \u05d4\u05de\u05d6\u05e8\u05d7\u05d9 \u05d1\u05e9\u05d1\u05d5\u05e2 \u05d4\u05e8\u05d0\u05e9\u05d5\u05df \u05e9\u05dc \u05de\u05d0\u05d9. "
    "\u05d4\u05dd \u05e8\u05e9\u05de\u05d5 \u05db\u05dc \u05de\u05e2\u05d9\u05d9\u05df, \u05d0\u05ea \u05d4\u05e1\u05e4\u05d9\u05e7\u05d4 \u05e9\u05dc\u05d5 \u05d1\u05dc\u05d9\u05d8\u05e8\u05d9\u05dd \u05dc\u05d3\u05e7\u05d4 \u05d5\u05d0\u05ea \u05e6\u05d1\u05e2 \u05d4\u05de\u05d9\u05dd. "
    "\u05e8\u05d5\u05d1 \u05d4\u05de\u05e2\u05d9\u05d9\u05e0\u05d5\u05ea \u05d4\u05d9\u05d5 \u05e6\u05dc\u05d5\u05dc\u05d9\u05dd.",
    "\u05d1\u05e8\u05db\u05e1 \u05d9\u05e9 41 \u05de\u05e2\u05d9\u05d9\u05e0\u05d5\u05ea. 36 \u05de\u05d4\u05dd \u05d6\u05d5\u05e8\u05de\u05d9\u05dd \u05db\u05dc \u05d4\u05e9\u05e0\u05d4. "
    "\u05d4\u05e6\u05d5\u05d5\u05ea \u05d9\u05d7\u05d6\u05d5\u05e8 \u05d1\u05d0\u05d5\u05e7\u05d8\u05d5\u05d1\u05e8 \u05db\u05d3\u05d9 \u05dc\u05de\u05d3\u05d5\u05d3 \u05e9\u05d5\u05d1 \u05d0\u05ea \u05d0\u05d5\u05ea\u05dd \u05de\u05e2\u05d9\u05d9\u05e0\u05d5\u05ea \u05d1\u05e1\u05d5\u05e3 \u05d4\u05e2\u05d5\u05e0\u05d4 \u05d4\u05d9\u05d1\u05e9\u05d4.",
]


def hebrew() -> Image.Image:
    """A right-to-left page in two columns: the right column reads first."""
    img, d = page()
    title = font(HEBREW, 60)
    d.text((W - MARGIN, 220), HEBREW_TITLE, font=title, fill="black", anchor="ra", direction="rtl")
    body = font(HEBREW, 36)
    gutter = 80
    col_w = (W - 2 * MARGIN - gutter) // 2
    rights = [W - MARGIN, W - MARGIN - col_w - gutter]
    for col, text in enumerate(HEBREW_TEXT):
        y = 360
        for line in wrap(d, text, body, col_w):
            d.text((rights[col], y), line, font=body, fill="black", anchor="ra", direction="rtl")
            y += 56
    return img


VERTICAL_TEXT = (
    "\u8c03\u67e5\u961f\u5728\u4e94\u6708\u7b2c\u4e00\u5468\u8d70\u8bbf\u4e86\u4e1c\u5cad\u3002"
    "\u4ed6\u4eec\u8bb0\u5f55\u4e86\u6bcf\u4e00\u5904\u6cc9\u6c34\u7684\u6d41\u91cf\u548c\u6c34\u7684\u989c\u8272\u3002"
    "\u5927\u591a\u6570\uff0c\u6cc9\u6c34\u6e05\u6f88\u96e8\u540e\u6709\u4e24\u5904\u53d8\u6210\u8910\u8272\u3002"
    "\u7b2c\u4e8c\u5929\u8fd9\u4e24\u5904\u6cc9\u6c34\u53c8\u6062\u590d\u6e05\u6f88"
    "\u4e1c\u5cad\u5171\u6709\u56db\u5341\u4e00\u5904\u6cc9\u6c34\u5176\u4e2d\u4e09\u5341\u516d\u5904\u5e38\u5e74\u6709\u6c34"
    "\u8c03\u67e5\u961f\u5c06\u5728\u5341\u6708\u518d\u6b21\u6d4b\u91cf\u8fd9\u4e9b\u6cc9\u6c34"
) * 3


def vertical() -> Image.Image:
    """Chinese set in vertical columns, read from the right column to the left.
    Songti, wide column gaps and full-stop marks: the orientation model calls
    this layout vertical (0.965); tighter columns read as a page turned sideways."""
    img, d = page()
    size = 44
    f = font(str(SUPP / "Songti.ttc"), size)
    x, i = W - MARGIN - size, 0
    while x > MARGIN:
        y = 200
        for _ in range(36):
            if y > H - 200:
                break
            d.text((x, y), VERTICAL_TEXT[i % len(VERTICAL_TEXT)], font=f, fill="black")
            i += 1
            y += int(size * 1.05)
        x -= int(size * 2.2)
    return img


def unicode_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    pdfmetrics.registerFont(TTFont("ArialUnicode", UNICODE_TTF))
    return "ArialUnicode"


def sideways_pdf(path: Path, turned: Path) -> None:
    """``path``: a table drawn a quarter turn sideways on a portrait page.
    ``turned``: upright content on a page turned by /Rotate 90."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    pw, ph = letter
    rows = [
        ["Spring", "Slope", "Flow (L/min)", "Colour"],
        ["North Gate", "North", "42.5", "Clear"],
        ["Fern Hollow", "East", "37.0", "Clear"],
        ["Red Stone", "South", "21.8", "Brown"],
        ["Old Mill", "West", "19.4", "Clear"],
    ]
    c.setFont("Helvetica", 9)
    c.drawString(72, 40, "Appendix B")
    c.saveState()
    c.translate(pw - 120, 120)
    c.rotate(90)
    c.setFont("Times-Bold", 16)
    c.drawString(0, 0, "Table 2. Flow of the four largest springs, May 2026.")
    for r, row in enumerate(rows):
        c.setFont("Times-Bold" if r == 0 else "Times-Roman", 13)
        for k, cell in enumerate(row):
            c.drawString(k * 130, -36 - 24 * r, cell)
    c.restoreState()
    c.showPage()
    c.save()
    c = canvas.Canvas(str(turned), pagesize=letter, invariant=1)
    c.setPageRotation(90)
    c.setFont("Times-Bold", 20)
    c.drawString(72, ph - 90, "Springs of the Eastern Ridge")
    c.setFont("Times-Roman", 12)
    y = ph - 130
    for _, text in COLUMN_TEXT[:2]:
        for line in wrap_pdf(c, text, "Times-Roman", 12, pw - 144):
            c.drawString(72, y, line)
            y -= 16
        y -= 10
    c.showPage()
    c.save()


def wrap_pdf(c, text: str, face: str, size: float, width: float) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, face, size) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def rtl_text_pdf(path: Path) -> None:
    """Hebrew in the text layer, drawn in visual order: pdfium reads it back in
    logical order, so the layer is trusted."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    face = unicode_font()
    c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    pw, ph = letter
    c.setFont(face, 20)
    c.drawRightString(pw - 72, ph - 90, HEBREW_TITLE[::-1])
    c.setFont(face, 12)
    y = ph - 130
    for text in HEBREW_TEXT:
        for line in wrap_pdf(c, text, face, 12, pw - 144):
            c.drawRightString(pw - 72, y, line[::-1])
            y -= 18
        y -= 10
    c.showPage()
    c.save()


def vertical_pdf(path: Path) -> None:
    """Chinese in vertical columns in the text layer, right column first."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    face = unicode_font()
    c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    pw, ph = letter
    x = pw - 90
    for i, line in enumerate(CJK_LINES[:7]):
        size = 20 if i == 0 else 15
        c.setFont(face, size)
        y = ph - 90
        for ch in line:
            c.drawString(x, y, ch)
            y -= size * 1.15
        x -= size * 1.9
    c.showPage()
    c.save()


LEGACY_LINES = [
    # (Devanagari as the page shows it, Latin codes as a legacy font stores them)
    (
        "\u0938\u0930\u094d\u0935\u0947\u0915\u094d\u0937\u0923 \u0926\u0932 \u0928\u0947 \u092a\u0942\u0930\u094d\u0935\u0940 \u092a\u0939\u093e\u0921\u093c\u0940 \u0915\u093e \u0926\u094c\u0930\u093e \u0915\u093f\u092f\u093e\u0964",
        "lo{k.k ny us iwohZ igkM+h dk nkSjk fd;kA",
    ),
    (
        "\u0905\u0927\u093f\u0915\u093e\u0902\u0936 \u091d\u0930\u0928\u0947 \u0938\u093e\u092b \u0925\u0947\u0964",
        "vf/kdka\"k >jus lkQ FksA",
    ),
    (
        "\u0926\u0932 \u0905\u0915\u094d\u091f\u0942\u092c\u0930 \u092e\u0947\u0902 \u092b\u093f\u0930 \u0938\u0947 \u092e\u093e\u092a \u0932\u0947\u0917\u093e\u0964",
        "ny vDVwcj esa fQj ls eki ysxkA",
    ),
    (
        "\u092a\u0942\u0930\u094d\u0935\u0940 \u092a\u0939\u093e\u0921\u093c\u0940 \u092a\u0930 \u0907\u0915\u0924\u093e\u0932\u0940\u0938 \u091d\u0930\u0928\u0947 \u0939\u0948\u0902\u0964",
        "iwohZ igkM+h ij bdrkyhl >jus gSaA",
    ),
]


def legacy_font_pdf(path: Path) -> None:
    """A Hindi page whose text layer stores Latin codes, as legacy fonts
    (Kruti Dev, DevLys) do: the page shows Devanagari, the layer reads as
    Latin. Each line's Latin text is drawn first, then an image of the
    Devanagari line covers it, so the layer sits under real ink."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    pw, ph = letter
    f = font(HINDI, 96)
    y = ph - 110
    for shown, codes in LEGACY_LINES:
        c.setFont("Times-Roman", 16)
        c.drawString(72, y, codes)
        w = c.stringWidth(codes, "Times-Roman", 16)
        img = Image.new("L", (int(f.getlength(shown)) + 20, 150), 255)
        ImageDraw.Draw(img).text((10, 10), shown, font=f, fill=0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        h = w * img.height / img.width
        c.drawImage(ImageReader(buf), 72, y - 0.3 * h, width=w, height=h)
        y -= 48
    c.showPage()
    c.save()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = two_column()
    pages = {
        "two-column.png": cols,
        "rotated-90.png": cols.rotate(90, expand=True, fillcolor=255),
        "rotated-180.png": cols.rotate(180, expand=True, fillcolor=255),
        "table.png": table(),
        "chinese.png": chinese(),
        "math.png": math_page(),
        "multiscript.png": multiscript(),
        "hebrew.png": hebrew(),
        "vertical.png": vertical(),
        "tilted-30.png": cols.resize((W * 3 // 5, H * 3 // 5), Image.LANCZOS).rotate(
            30, expand=True, fillcolor=255
        ),
    }
    for name, img in pages.items():
        img.save(OUT / name, optimize=True)
    digital_pdf(OUT / "digital.pdf")
    scanned_pdf(OUT / "scanned.pdf", cols)
    locked_pdf(OUT / "locked.pdf")
    sideways_pdf(OUT / "sideways.pdf", OUT / "rotate-90.pdf")
    rtl_text_pdf(OUT / "rtl-text.pdf")
    vertical_pdf(OUT / "vertical-text.pdf")
    legacy_font_pdf(OUT / "legacy-font.pdf")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}\t{p.stat().st_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
