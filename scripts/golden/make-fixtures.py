"""Generate the parity fixtures in tests/fixtures/pages/.

Run by hand from the repo root:

    uvx --with pillow --with reportlab --with matplotlib python -I scripts/golden/make-fixtures.py

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
    }
    for name, img in pages.items():
        img.save(OUT / name, optimize=True)
    digital_pdf(OUT / "digital.pdf")
    scanned_pdf(OUT / "scanned.pdf", cols)
    locked_pdf(OUT / "locked.pdf")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name}\t{p.stat().st_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
