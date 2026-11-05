"""
report_pdf.py - NSW Defect Detective diagnostic report renderer.

Drop-in replacement for the old fpdf2 `generate_pdf_report`.

Why not fpdf2 + write_html?
  * fpdf2's HTML subset chokes on <thead>/<tbody> produced by python-markdown
    and throws "Invalid HTML: <th> used outside any <tr>", which sent every AI
    message into the ASCII fallback (that is why the PDF had no colour).
  * fpdf2's core fonts are latin-1 only, so "°C", "µm", "≥" or an emoji raised
    FPDFUnicodeEncodingException and killed the whole export.
  * fpdf2 cannot wrap long table cells sensibly.

This module parses Markdown itself and emits ReportLab Platypus flowables, so
tables wrap, colours are exact, and any Unicode the model emits is safe.

Public API
----------
    build_report_pdf(serial, messages, logo_path=None, platform=None,
                     pump=None, generated_at=None) -> bytes

`messages` is the Streamlit chat list: [{"role": "user"|"assistant",
"content": "..."}].
"""
from __future__ import annotations

import datetime as _dt
import html as _html
import os
import re
import unicodedata
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image, KeepTogether, ListFlowable,
    ListItem, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

# --------------------------------------------------------------------------- #
# NSW brand palette
#
# The two brand colours are exactly the ones used in app.py
#   #14213D  "NSW Dark Blue"   (pdf.set_text_color(20, 33, 61))
#   #FCA311  "NSW Orange"      (pdf.set_text_color(252, 163, 17))
# Everything else is derived from them. Edit PALETTE and the whole report restyles -
# nothing downstream hard-codes a colour.
# --------------------------------------------------------------------------- #
PALETTE = {
    "navy":        "#14213D",   # NSW Dark Blue - headings, table header, header band
    "orange":      "#FCA311",   # NSW Orange - rules, accents, operator marker
    "navy_light":  "#25406E",   # sub-headings
    "cream":       "#FFF5E6",   # callout fill
    "operator_bg": "#FFF9EF",   # operator message panel
    "grey":        "#5A5F6E",   # captions, footer
    "rule":        "#D9DCE3",   # hairlines and grid
    "zebra":       "#F5F6F8",   # alternating table rows
    "band_text":   "#B9C2D6",   # subtitle on the navy band
    "code":        "#B3261E",   # inline `code`
    "link":        "#1B4F9C",   # external links
    # Confidence ratings - keyed by the word the model writes in the table
    "high":        "#1B7F3B",
    "medium":      "#B26A00",
    "low":         "#B3261E",
}


def C(name: str):
    """Look up a palette colour as a ReportLab Color."""
    return colors.HexColor(PALETTE[name])


NSW_NAVY = C("navy")
NSW_ORANGE = C("orange")
NSW_CREAM = C("cream")
NSW_GREY = C("grey")
NSW_RULE = C("rule")
ZEBRA = C("zebra")

CONF_COLOURS = {"high": C("high"), "medium": C("medium"), "low": C("low")}

# --------------------------------------------------------------------------- #
# Fonts - must be Unicode-capable or "°C"/"µm"/"≥" become black boxes
# --------------------------------------------------------------------------- #
_FONT_CANDIDATES = [
    # (regular, bold, italic, bold-italic)
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"),
    ("C:/Windows/Fonts/DejaVuSans.ttf", "C:/Windows/Fonts/DejaVuSans-Bold.ttf",
     "C:/Windows/Fonts/DejaVuSans-Oblique.ttf",
     "C:/Windows/Fonts/DejaVuSans-BoldOblique.ttf"),
    ("/Library/Fonts/DejaVuSans.ttf", "/Library/Fonts/DejaVuSans-Bold.ttf",
     "/Library/Fonts/DejaVuSans-Oblique.ttf",
     "/Library/Fonts/DejaVuSans-BoldOblique.ttf"),
]

BODY_FONT = "NSWSans"
BODY_BOLD = "NSWSans-Bold"
MONO_FONT = "Courier"

_FONTS_READY = False


def _bundled_vera() -> tuple[str, str, str, str]:
    """reportlab always ships the Bitstream Vera family - a guaranteed fallback."""
    import reportlab
    d = os.path.join(os.path.dirname(reportlab.__file__), "fonts")
    return (os.path.join(d, "Vera.ttf"), os.path.join(d, "VeraBd.ttf"),
            os.path.join(d, "VeraIt.ttf"), os.path.join(d, "VeraBI.ttf"))


def _register_fonts() -> None:
    global _FONTS_READY, MONO_FONT
    if _FONTS_READY:
        return
    for quad in _FONT_CANDIDATES + [_bundled_vera()]:
        if all(os.path.exists(p) for p in quad):
            try:
                pdfmetrics.registerFont(TTFont(BODY_FONT, quad[0]))
                pdfmetrics.registerFont(TTFont(BODY_BOLD, quad[1]))
                pdfmetrics.registerFont(TTFont(BODY_FONT + "-It", quad[2]))
                pdfmetrics.registerFont(TTFont(BODY_FONT + "-BdIt", quad[3]))
                pdfmetrics.registerFontFamily(
                    BODY_FONT, normal=BODY_FONT, bold=BODY_BOLD,
                    italic=BODY_FONT + "-It", boldItalic=BODY_FONT + "-BdIt")
                mono = os.path.join(os.path.dirname(quad[0]), "DejaVuSansMono.ttf")
                if os.path.exists(mono):
                    pdfmetrics.registerFont(TTFont("NSWMono", mono))
                    MONO_FONT = "NSWMono"
                _FONTS_READY = True
                return
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
    _FONTS_READY = True  # fall back to Helvetica; still renders, just latin-1


# --------------------------------------------------------------------------- #
# Text hygiene
# --------------------------------------------------------------------------- #
# Emoji / pictographs have no glyph in any of the above fonts -> black boxes.
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002190-\U000021FF\U00002600-\U000027BF"
    "\U0000FE00-\U0000FE0F\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF]+"
)
_EMOJI_WORDS = {
    "📊": "[telemetry]", "🟢": "[OK]", "🟡": "[WARN]", "🔴": "[CRITICAL]",
    "⚪": "[N/A]", "🔬": "", "🏭": "", "✅": "[done]", "⚠": "[warning]",
    "🎙": "", "📄": "", "📧": "", "🔄": "", "📷": "", "⬅": "", "🎛": "",
}


def clean_text(text: str) -> str:
    """Make any model/operator string safe for an embedded TTF."""
    if not text:
        return ""
    text = text.replace("\u200b", "").replace("\ufeff", "")
    for emo, word in _EMOJI_WORDS.items():
        text = text.replace(emo, word)
    text = _EMOJI_RE.sub("", text)
    # Normalise so composed accents render with a single glyph.
    text = unicodedata.normalize("NFC", text)
    return text


# --------------------------------------------------------------------------- #
# Markdown repair - fixes the model's most common layout mistakes
# --------------------------------------------------------------------------- #
_LINK_RE = re.compile(r'\[([^\]\n]*)\]\(([^()\s]*)(?:\s+"([^"]*)")?\)')


def repair_markdown(md: str) -> str:
    """Fix LLM table output before anything tries to render it.

    1. An unescaped '|' inside a link title (the prompt's own example has one:
       `[1](# "Path: x.pdf | Page: 4")`) splits the Markdown cell and shifts
       every later column. We swap it for an en dash.
    2. Rows with the wrong number of cells are padded / trimmed to the header.
    3. A missing separator row after the header is inserted.
    """
    md = clean_text(md)

    # 1. pipes inside link titles / parentheses
    def _fix_link(m: re.Match) -> str:
        label, target, title = m.group(1), m.group(2), m.group(3)
        if title:
            title = title.replace("|", "-")
            return f'[{label}]({target} "{title}")'
        return m.group(0)

    md = _LINK_RE.sub(_fix_link, md)

    lines = md.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_table_row(line):
            block, i = _collect_table(lines, i)
            out.extend(_normalise_table(block))
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _is_separator(line: str) -> bool:
    return bool(re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", line)) and "-" in line


def _collect_table(lines: list[str], start: int) -> tuple[list[str], int]:
    i = start
    block: list[str] = []
    while i < len(lines) and _is_table_row(lines[i]):
        block.append(lines[i])
        i += 1
    return block, i


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    # honour escaped pipes
    cells = re.split(r"(?<!\\)\|", s)
    return [c.replace("\\|", "|").strip() for c in cells]


def _normalise_table(block: list[str]) -> list[str]:
    rows = [_split_row(l) for l in block]
    seps = [i for i, l in enumerate(block) if _is_separator(l)]
    header = rows[0]
    width = len(header)
    fixed = [header]
    if not seps:
        fixed.append(["---"] * width)
    for idx, row in enumerate(rows[1:], 1):
        if idx in seps:
            fixed.append(["---"] * width)
            continue
        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            # merge the overflow back into the last kept column
            row = row[: width - 1] + [" ".join(row[width - 1:])]
        fixed.append(row)
    return ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in fixed]


# --------------------------------------------------------------------------- #
# Inline Markdown -> ReportLab mini-XML
# --------------------------------------------------------------------------- #
_CODE_RE = re.compile(r"`([^`]+)`")


def inline_md(text: str, collect_sources: list | None = None) -> str:
    """Convert **bold**, *italic*, `code` and [n](# "Path: ...") to RL markup."""
    text = clean_text(text)

    # Protect code spans first so ** inside them survives.
    spans: list[str] = []

    def _stash(m: re.Match) -> str:
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = _CODE_RE.sub(_stash, text)
    text = _html.escape(text, quote=False)

    def _link(m: re.Match) -> str:
        label = m.group(1) or "ref"
        target, title = m.group(2) or "", m.group(3) or ""
        if collect_sources is not None and title:
            if title not in collect_sources:
                collect_sources.append(title)
            n = collect_sources.index(title) + 1
            return (f'<font color="{PALETTE["navy"]}"><b>{label}</b></font>'
                    f'<super><font size="6" color="{PALETTE["orange"]}">{n}</font></super>')
        if target.startswith(("http://", "https://")):
            return f'<link href="{target}" color="{PALETTE["link"]}"><u>{label}</u></link>'
        return f"<b>{label}</b>"

    text = _LINK_RE.sub(_link, text)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<strike>\1</strike>", text)

    def _unstash(m: re.Match) -> str:
        raw = _html.escape(spans[int(m.group(1))], quote=False)
        return f'<font face="{MONO_FONT}" size="8.5" color="{PALETTE["code"]}">{raw}</font>'

    return re.sub(r"\x00(\d+)\x00", _unstash, text)


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def _styles() -> dict:
    _register_fonts()
    base = getSampleStyleSheet()
    font = BODY_FONT if BODY_FONT in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold = BODY_BOLD if BODY_BOLD in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"

    def s(name, **kw):
        kw.setdefault("fontName", font)
        kw.setdefault("fontSize", 9.5)
        kw.setdefault("leading", 13.5)
        kw.setdefault("textColor", colors.black)
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    return {
        "font": font, "bold": bold,
        "title": s("t", fontName=bold, fontSize=19, leading=23,
                   textColor=NSW_NAVY, alignment=TA_LEFT, spaceAfter=2),
        "subtitle": s("st", fontSize=9, textColor=NSW_GREY, spaceAfter=10),
        "h1": s("h1", fontName=bold, fontSize=13.5, leading=17,
                textColor=NSW_NAVY, spaceBefore=12, spaceAfter=5),
        "h2": s("h2", fontName=bold, fontSize=11, leading=14,
                textColor=C("navy_light"), spaceBefore=9, spaceAfter=3),
        "h3": s("h3", fontName=bold, fontSize=10, leading=13,
                textColor=NSW_GREY, spaceBefore=7, spaceAfter=2),
        "body": s("b", spaceAfter=5),
        "li": s("li", spaceAfter=2, leading=13),
        "cell": s("c", fontSize=8.5, leading=11.5),
        "cellb": s("cb", fontName=bold, fontSize=8.5, leading=11.5),
        "cellc": s("cc", fontName=bold, fontSize=8.5, leading=11.5, alignment=TA_CENTER),
        "th": s("th", fontName=bold, fontSize=8.5, leading=11.5, textColor=NSW_ORANGE),
        "quote": s("q", fontSize=8.5, leading=12, textColor=NSW_GREY,
                   leftIndent=8, spaceAfter=5),
        "role_op": s("ro", fontName=bold, fontSize=9.5, textColor=NSW_ORANGE,
                     spaceBefore=2, spaceAfter=3),
        "role_ai": s("ra", fontName=bold, fontSize=9.5, textColor=NSW_NAVY,
                     spaceBefore=2, spaceAfter=3),
        "src": s("src", fontSize=7.5, leading=10, textColor=NSW_GREY),
    }


# --------------------------------------------------------------------------- #
# Block Markdown -> flowables
# --------------------------------------------------------------------------- #
def md_to_flowables(md: str, st: dict, avail_width: float,
                    sources: list | None = None) -> list:
    md = repair_markdown(md)
    lines = md.split("\n")
    flow: list = []
    buf: list[str] = []
    list_buf: list[tuple[str, str]] = []

    def flush_para():
        if buf:
            txt = " ".join(x.strip() for x in buf).strip()
            if txt:
                flow.append(Paragraph(inline_md(txt, sources), st["body"]))
            buf.clear()

    def flush_list():
        if not list_buf:
            return
        kind = list_buf[0][0]
        items = [ListItem(Paragraph(inline_md(t, sources), st["li"]), leftIndent=14)
                 for _, t in list_buf]
        flow.append(ListFlowable(
            items, bulletType="1" if kind == "ol" else "bullet",
            bulletFontName=st["font"], bulletFontSize=8.5,
            bulletColor=NSW_ORANGE, leftIndent=16, spaceAfter=6))
        list_buf.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_para(); flush_list()
            i += 1
            continue

        if re.match(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", stripped):
            flush_para(); flush_list()
            flow.append(Spacer(1, 3))
            flow.append(HRFlowable(width="100%", thickness=0.6, color=NSW_RULE))
            flow.append(Spacer(1, 5))
            i += 1
            continue

        h = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if h:
            flush_para(); flush_list()
            level, text = len(h.group(1)), h.group(2)
            key = "h1" if level <= 3 else ("h2" if level == 4 else "h3")
            flow.append(Paragraph(inline_md(text, sources), st[key]))
            if key == "h1":
                flow.append(HRFlowable(width="100%", thickness=1.4,
                                       color=NSW_ORANGE, spaceAfter=5))
            i += 1
            continue

        if _is_table_row(line):
            flush_para(); flush_list()
            block, i = _collect_table(lines, i)
            tbl = _build_table(block, st, avail_width, sources)
            if tbl is not None:
                flow.append(Spacer(1, 2))
                flow.append(tbl)
                flow.append(Spacer(1, 7))
            continue

        m = re.match(r"^\s*(\d+)[.)]\s+(.*)$", line)
        if m:
            flush_para()
            if list_buf and list_buf[0][0] != "ol":
                flush_list()
            list_buf.append(("ol", m.group(2)))
            i += 1
            continue

        m = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if m:
            flush_para()
            if list_buf and list_buf[0][0] != "ul":
                flush_list()
            list_buf.append(("ul", m.group(1)))
            i += 1
            continue

        if stripped.startswith(">"):
            flush_para(); flush_list()
            quote = re.sub(r"^\s*>\s?", "", line)
            flow.append(_callout(inline_md(quote, sources), st, avail_width))
            i += 1
            continue

        flush_list()
        buf.append(line)
        i += 1

    flush_para(); flush_list()
    return flow


def _callout(markup: str, st: dict, width: float) -> Table:
    p = Paragraph(markup, st["quote"])
    t = Table([[p]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NSW_CREAM),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, NSW_ORANGE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _conf_colour(value: str):
    v = value.strip().lower()
    for key, col in CONF_COLOURS.items():
        if key in v:
            return col
    m = re.search(r"(\d{1,3})\s*%?", v)
    if m:
        n = int(m.group(1))
        if n >= 70:
            return CONF_COLOURS["high"]
        if n >= 40:
            return CONF_COLOURS["medium"]
        return CONF_COLOURS["low"]
    return None


def _plain(cell: str) -> str:
    """Approximate what a cell will actually look like once rendered, so column
    widths are based on visible text - not on the 50 characters of link markup
    that `[1](# "Path: ... Page: 4")` contributes and which never gets printed."""
    text = _LINK_RE.sub(lambda m: m.group(1) or "", cell)
    text = re.sub(r"[*_`~]", "", text)
    return text.strip()


def _column_widths(header, body, ncols, avail, st) -> list[float]:
    font = st["font"]
    pad = 13.0
    cols = [[_plain(header[c])] + [_plain(r[c]) for r in body if c < len(r)]
            for c in range(ncols)]

    # Hard minimum: the longest unbreakable word must fit, else ReportLab
    # hyphen-chops it mid-token ("dispenser_S 350.pdf").
    bold = st["bold"]
    mins = []
    for texts in cols:
        longest = 0.0
        for idx, t in enumerate(texts):
            face = bold if idx == 0 else font       # row 0 is the header
            for word in (t.split() or [""]):
                longest = max(longest, pdfmetrics.stringWidth(word, face, 8.5))
        mins.append(min(longest + pad, avail * 0.30))

    weights = [max(4.0, sum(len(t) for t in texts) / max(1, len(texts)))
               for texts in cols]
    total = sum(weights) or 1.0
    widths = [max(avail * w / total, mins[i]) for i, w in enumerate(weights)]

    overflow = sum(widths) - avail
    if overflow > 0:
        slack = [max(0.0, widths[i] - mins[i]) for i in range(ncols)]
        pool = sum(slack)
        if pool > 0:
            take = min(overflow, pool)
            widths = [widths[i] - take * slack[i] / pool for i in range(ncols)]
        else:
            k = avail / sum(widths)
            widths = [w * k for w in widths]
    else:
        widths = [w + (-overflow) * w / sum(widths) for w in widths]
    return widths


def _build_table(block: list[str], st: dict, avail: float, sources) -> Table | None:
    rows = [_split_row(l) for l in block if not _is_separator(l)]
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 1:
        return None
    header, body = rows[0], rows[1:]
    ncols = len(header)

    widths = _column_widths(header, body, ncols, avail, st)

    conf_col = next((i for i, h in enumerate(header)
                     if re.search(r"confidence|certainty", h, re.I)), None)

    data = [[Paragraph(inline_md(h, sources), st["th"]) for h in header]]
    for r in body:
        cells = []
        for c in range(ncols):
            raw = r[c] if c < len(r) else ""
            if c == conf_col:
                col = _conf_colour(raw)
                style = ParagraphStyle("cx", parent=st["cellc"],
                                       textColor=col or colors.black)
                cells.append(Paragraph(inline_md(raw, sources), style))
            elif c == 0:
                cells.append(Paragraph(inline_md(raw, sources), st["cellb"]))
            else:
                cells.append(Paragraph(inline_md(raw, sources), st["cell"]))
        data.append(cells)

    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NSW_NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, NSW_RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 1.2, NSW_ORANGE),
        ("BOX", (0, 0), (-1, -1), 0.8, NSW_NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for ri in range(1, len(data)):
        if ri % 2 == 0:
            style.append(("BACKGROUND", (0, ri), (-1, ri), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
class _Doc(BaseDocTemplate):
    def __init__(self, buf, serial, subtitle, logo_path, footer_note=None,
                 doc_title=None, **kw):
        super().__init__(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                         topMargin=30 * mm, bottomMargin=18 * mm,
                         title=doc_title or f"NSW Diagnostic Report {serial}",
                         author="NSW Automation", **kw)
        self.serial, self.subtitle, self.logo_path = serial, subtitle, logo_path
        # footer_note=None keeps the legacy disclaimer on every page.
        # footer_note="" prints the serial and page number only - the disclaimer then lives
        # once, under the header block, instead of repeating on every page.
        self.footer_note = footer_note
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      id="body")
        self.addPageTemplates([PageTemplate(id="nsw", frames=[frame],
                                            onPage=self._decorate)])

    def _decorate(self, canv, doc):
        _register_fonts()
        font = BODY_FONT if BODY_FONT in pdfmetrics.getRegisteredFontNames() else "Helvetica"
        bold = BODY_BOLD if BODY_BOLD in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
        w, h = A4
        canv.saveState()
        # header band
        canv.setFillColor(NSW_NAVY)
        canv.rect(0, h - 22 * mm, w, 22 * mm, stroke=0, fill=1)
        canv.setFillColor(NSW_ORANGE)
        canv.rect(0, h - 23 * mm, w, 1 * mm, stroke=0, fill=1)

        # The NSW logo sits on a white card. Most logo artwork is dark-on-transparent,
        # so dropping it straight onto the navy band would make it invisible; the card
        # guarantees it reads regardless of how the PNG was authored.
        drew_logo = False
        if self.logo_path and os.path.exists(self.logo_path):
            try:
                iw, ih = ImageReader(self.logo_path).getSize()
                box_h = 13.5 * mm
                box_w = min(46 * mm, max(20 * mm, box_h * (iw / float(ih or 1)) + 5 * mm))
                x, y = 14 * mm, h - 18.25 * mm
                canv.setFillColor(colors.white)
                canv.roundRect(x, y, box_w, box_h, 1.6 * mm, stroke=0, fill=1)
                canv.drawImage(self.logo_path, x + 2.5 * mm, y + 1.75 * mm,
                               width=box_w - 5 * mm, height=box_h - 3.5 * mm,
                               preserveAspectRatio=True, anchor="c", mask="auto")
                drew_logo = True
            except Exception:  # noqa: BLE001 - a bad image must never break the report
                drew_logo = False
        if not drew_logo:
            canv.setFillColor(NSW_ORANGE)
            canv.setFont(bold, 13)
            canv.drawString(16 * mm, h - 14 * mm, "NSW AUTOMATION")

        canv.setFillColor(colors.white)
        canv.setFont(bold, 10)
        canv.drawRightString(w - 16 * mm, h - 12 * mm, "AI Defect Detective")
        canv.setFont(font, 7.5)
        canv.setFillColor(C("band_text"))
        canv.drawRightString(w - 16 * mm, h - 17 * mm, self.subtitle)

        # footer
        canv.setStrokeColor(NSW_RULE)
        canv.setLineWidth(0.5)
        canv.line(16 * mm, 13 * mm, w - 16 * mm, 13 * mm)
        canv.setFont(font, 7.5)
        canv.setFillColor(NSW_GREY)
        default_note = ("AI-generated; verify against NSW manuals before acting.")
        note = default_note if self.footer_note is None else self.footer_note
        canv.drawString(16 * mm, 9 * mm,
                        f"Machine {self.serial}" + (f"  -  {note}" if note else ""))
        canv.drawRightString(w - 16 * mm, 9 * mm, f"Page {canv.getPageNumber()}")
        canv.restoreState()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_report_pdf(serial: str, messages, logo_path: str | None = None,
                     platform: str | None = None, pump: str | None = None,
                     generated_at: _dt.datetime | None = None) -> bytes:
    _register_fonts()
    st = _styles()
    stamp = (generated_at or _dt.datetime.now()).strftime("%Y-%m-%d %H:%M")
    subtitle = " | ".join(x for x in [platform, pump, stamp] if x)

    buf = BytesIO()
    doc = _Doc(buf, serial, subtitle, logo_path)
    avail = doc.width
    sources: list[str] = []

    story: list = [
        Paragraph(f"Diagnostic Report &nbsp;<font color='{PALETTE['orange']}'>"
                  f"{_html.escape(serial)}</font>",
                  st["title"]),
        Paragraph(_html.escape(subtitle or ""), st["subtitle"]),
        HRFlowable(width="100%", thickness=1.4, color=NSW_NAVY, spaceAfter=10),
    ]

    for msg in messages or []:
        role = (msg.get("role") or "assistant").lower()
        content = msg.get("content") or ""
        if not content.strip():
            continue
        if role == "user":
            header = Paragraph("OPERATOR", st["role_op"])
            body = md_to_flowables(content, st, avail - 6 * mm, sources)
            inner = Table([[[header] + body]], colWidths=[avail])
            inner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), C("operator_bg")),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, NSW_ORANGE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(inner)
        else:
            story.append(Paragraph("AI DEFECT DETECTIVE", st["role_ai"]))
            story.extend(md_to_flowables(content, st, avail, sources))
        story.append(Spacer(1, 9))

    if sources:
        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=0.8, color=NSW_RULE, spaceAfter=6))
        story.append(Paragraph("Cited sources", st["h2"]))
        for n, src in enumerate(sources, 1):
            story.append(Paragraph(
                f'<super><font color="{PALETTE["orange"]}">{n}</font></super> '
                f'&nbsp;{_html.escape(src)}',
                st["src"]))

    try:
        doc.build(story)
    except Exception:  # noqa: BLE001 - never let the export kill the app
        buf = BytesIO()
        doc = _Doc(buf, serial, subtitle, logo_path)
        plain = [Paragraph(_html.escape(clean_text(m.get("content", ""))).replace("\n", "<br/>"),
                           st["body"]) for m in (messages or [])]
        doc.build(plain or [Paragraph("No content.", st["body"])])
    return buf.getvalue()


# Backwards-compatible alias for the old call site in app.py.
def generate_pdf_report(serial, messages, logo_path=None, **kw) -> bytes:
    return build_report_pdf(serial, messages, logo_path=logo_path, **kw)
