#!/usr/bin/env python3
"""Portal-ready PDF mirror of build_sih_deck.py (PS 26153, SIH 2026).

Same 6 slides, same coordinates/content. Fonts:
  - Times-Bold  -> display titles (metric stand-in for Times New Roman)
  - DejaVu Sans -> body (covers all symbols; wider than Arial = conservative
    overflow check for the PPTX)
  - Helvetica-Oblique -> ASCII/WinAnsi-only italic footnotes

Run:  python3 tools/build_sih_deck_pdf.py
Out:  docs/SIH_PS26153_ARGUS_6slide.pdf   (+ /tmp/sih_png/slideN.png previews)
"""
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


class PP_ALIGN:  # python-pptx-style aliases
    LEFT, CENTER, RIGHT = TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import Paragraph
import os

# ---------------------------------------------------------------- palette
NAVY   = HexColor("#1F3864")
BLUE   = HexColor("#0E76BE")
DBLUE  = HexColor("#1F4E79")
BLACK  = HexColor("#1A1A1A")
GREY   = HexColor("#444444")
PURPLE = HexColor("#7C4DA0")
GREEN  = HexColor("#1E8E3E")
ORANGE = HexColor("#E67E22")
RED    = HexColor("#C0392B")
LBLUE  = HexColor("#EAF2FB")
LGREEN = HexColor("#ECF7EF")
LORNG  = HexColor("#FDF2E9")
LYELL  = HexColor("#FDF6E3")
LNAVY  = HexColor("#EAF0F8")
SAFF   = HexColor("#FF9933")
INKGRN = HexColor("#138808")
GOLD   = HexColor("#FFD54F")

SANS, SANSB = "DJS", "DJ"          # DejaVu Sans / Bold
SERIF, SERIFB = "Times-Roman", "Times-Bold"
OBL = "Helvetica-Oblique"

pdfmetrics.registerFont(TTFont(SANS, "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont(SANSB, "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFontFamily(SANS, normal=SANS, bold=SANSB, italic=SANS, boldItalic=SANSB)

W, H = 13.333 * 72, 7.5 * 72
FOOT_Y, FOOT_H = 7.15, 0.35
OUT = "/home/user/Argus/docs/SIH_PS26153_ARGUS_6slide.pdf"
PNG_DIR = "/tmp/sih_png"


# ---------------------------------------------------------------- helpers
def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def st(name, font=SANS, size=12, bold=False, color=BLACK, align=PP_ALIGN.LEFT,
       leading=None, space_after=0, underline=False, italic=False):
    if bold:
        font = SANSB if font in (SANS, SANSB) else font
    if italic:
        font = OBL if font in ("Helvetica", OBL) else font
    return ParagraphStyle(name, fontName=font, fontSize=size,
                          leading=leading or size * 1.22, textColor=color,
                          alignment=align, spaceAfter=space_after,
                          underline=underline)


def wrap_ph(html, style, w_pt):
    p = Paragraph(html, style)
    _, ph = p.wrap(w_pt, 10_000)
    return p, ph


def draw_in_box(c, html, style, x, y, w, h=None, anchor="top", margin_x=0.0,
                margin_y=0.0):
    """Draw a single Paragraph inside an inch-box (top-left origin)."""
    w_pt = (w - 2 * margin_x) * 72
    p, ph = wrap_ph(html, style, w_pt)
    if anchor == "middle":
        cy_pt = H - (y + h / 2) * 72
        y_pdf = cy_pt - ph / 2
    else:
        y_pdf = H - (y + margin_y) * 72 - ph
    p.drawOn(c, (x + margin_x) * 72, y_pdf)
    return ph / 72


def draw_stack(c, items, x, y, w, h=None, anchor="top", margin_x=0.0,
               margin_y=0.0):
    """items = [(html, style), ...] stacked with each style's spaceAfter."""
    w_pt = (w - 2 * margin_x) * 72
    phs, pobjs = [], []
    for html, style in items:
        p, ph = wrap_ph(html, style, w_pt)
        pobjs.append(p)
        phs.append(ph)
    total = sum(phs) + sum(s.spaceAfter for _, s in items[:-1])
    if anchor == "middle" and h:
        cy_pt = H - (y + h / 2) * 72
        cur = cy_pt + total / 2
    else:
        cur = H - (y + margin_y) * 72
    for p, ph, (_, style) in zip(pobjs, phs, items):
        cur -= ph
        p.drawOn(c, (x + margin_x) * 72, cur)
        cur -= style.spaceAfter
    return total / 72


def rrect(c, x, y, w, h, fill=None, stroke=None, lw=1.0, radius=None):
    if radius is None:
        radius = min(w, h) / 2 * 0.0  # square corners by default
    if fill is not None:
        c.setFillColor(fill)
    if stroke is not None:
        c.setStrokeColor(stroke)
        c.setLineWidth(lw)
    if radius:
        c.roundRect(x * 72, H - (y + h) * 72, w * 72, h * 72, radius * 72,
                    fill=1 if fill is not None else 0,
                    stroke=1 if stroke is not None else 0)
    else:
        c.rect(x * 72, H - (y + h) * 72, w * 72, h * 72,
               fill=1 if fill is not None else 0,
               stroke=1 if stroke is not None else 0)


def ellipse(c, x, y, w, h, fill=None, stroke=None, lw=1.0):
    if fill is not None:
        c.setFillColor(fill)
    if stroke is not None:
        c.setStrokeColor(stroke)
        c.setLineWidth(lw)
    c.ellipse(x * 72, H - (y + h) * 72, (x + w) * 72, H - y * 72,
              fill=1 if fill is not None else 0,
              stroke=1 if stroke is not None else 0)


def polygon(c, pts_in, fill):
    pth = c.beginPath()
    first = True
    for px, py in pts_in:
        if first:
            pth.moveTo(px * 72, py * 72)
            first = False
        else:
            pth.lineTo(px * 72, py * 72)
    pth.close()
    c.setFillColor(fill)
    c.drawPath(pth, fill=1, stroke=0)


def wordmark(c, x, y, w):
    draw_stack(c, [
        ("SMART INDIA", st("wm", SANSB, 8.5, True, NAVY, PP_ALIGN.RIGHT)),
        ("HACKATHON", st("wm2", SANSB, 8.5, True, NAVY, PP_ALIGN.RIGHT)),
        ("2026", st("wm3", SANSB, 12, True, INKGRN, PP_ALIGN.RIGHT)),
    ], x, y, w, h=0.8)


def tricolor_dots(c, y):
    for i, col in enumerate((SAFF, white, INKGRN)):
        ellipse(c, 12.55 + i * 0.16, y, 0.13, 0.13, fill=col,
                stroke=GREY, lw=0.5)


def header(c, title, page_no=None, team=True):
    draw_in_box(c, esc(title), ParagraphStyle(
        "t", fontName=SERIFB, fontSize=33, leading=39, textColor=BLACK,
        alignment=PP_ALIGN.CENTER), 1.75, 0.18, 9.8, 0.85, anchor="middle")
    if team:
        ellipse(c, 0.25, 0.12, 1.55, 0.8, fill=None, stroke=PURPLE, lw=1.5)
        draw_stack(c, [
            ("Your", st("tm", SANS, 11, False, BLACK, PP_ALIGN.CENTER)),
            ("Team Name", st("tm2", SANS, 11, False, BLACK, PP_ALIGN.CENTER)),
        ], 0.25, 0.12, 1.55, 0.8, anchor="middle")
    wordmark(c, 11.35, 0.14, 1.85)
    tricolor_dots(c, 0.78)
    rrect(c, 0, FOOT_Y, 13.333, FOOT_H, fill=BLUE)
    draw_in_box(c, "@SIH Idea submission- Template",
                st("ft", SANS, 11, False, white, PP_ALIGN.CENTER),
                0, FOOT_Y + 0.04, 13.333, 0.28)
    if page_no is not None:
        draw_in_box(c, str(page_no),
                    st("pn", SANSB, 11, True, white, PP_ALIGN.RIGHT),
                    12.583, FOOT_Y + 0.04, 0.5, 0.28)


def col_box(c, x, y, w, h, title, tcolor, body_fill, bullets,
            title_size=13, bullet_size=11.5, bullet_space=5):
    rrect(c, x, y, w, 0.44, fill=tcolor, stroke=tcolor, radius=0.079)
    draw_in_box(c, esc(title),
                st("ct", SANSB, title_size, True, white, PP_ALIGN.CENTER),
                x, y, w, 0.44, anchor="middle", margin_x=0.06)
    rrect(c, x, y + 0.40, w, h - 0.40, fill=body_fill, stroke=tcolor, lw=1.25,
          radius=0.16667 * min(w, h - 0.40))
    items = [(f"• {esc(b)}", st("cb", SANS, bullet_size, False, BLACK,
                                space_after=bullet_space)) for b in bullets]
    draw_stack(c, items, x, y + 0.40, w, h - 0.40,
               margin_x=0.12, margin_y=0.08)


def bullets_block(c, x, y, w, h, groups, size=12):
    items = []
    for gi, (label, lcolor, blist) in enumerate(groups):
        items.append((esc(label), st("lb", SANSB, size + 1, True, lcolor,
                                     space_after=3)))
        if gi > 0:
            items[-1][1].spaceBefore = 7
        for b in blist:
            items.append((f"• {esc(b)}", st("bl", SANS, size, False, BLACK,
                                            space_after=3)))
    # emulate group gap: add leading space via space_before on group labels
    cur_y = y
    first = True
    for gi, (label, lcolor, blist) in enumerate(groups):
        if not first:
            cur_y += 7 / 72
        first = False
        ph = draw_in_box(c, esc(label),
                         st("lb", SANSB, size + 1, True, lcolor,
                            space_after=3),
                         x, cur_y, w, margin_x=0)
        cur_y += ph + 3 / 72
        for b in blist:
            ph = draw_in_box(c, f"• {esc(b)}",
                             st("bl", SANS, size, False, BLACK, space_after=3),
                             x, cur_y, w, margin_x=0)
            cur_y += ph + 3 / 72
    return cur_y - y


def bullets_flow(c, x, y, w, groups, size=12, space=5, gap=0):
    """Continuous top-down bullet list (for col_box-like custom blocks)."""
    cur_y = y
    for label, lcolor, blist in groups:
        ph = draw_in_box(c, esc(label),
                         st("lb", SANSB, size + 1, True, lcolor), x, cur_y, w)
        cur_y += ph + 4 / 72
        for b in blist:
            ph = draw_in_box(c, f"• {esc(b)}",
                             st("bl", SANS, size, False, BLACK), x, cur_y, w)
            cur_y += ph + space / 72
        cur_y += gap / 72
    return cur_y - y


def strip(c, x, y, w, text):
    draw_in_box(c, esc(text), st("sp", SANSB, 12, True, GREY), x, y, w, 0.3)


# ================================================================ build
c = canvas.Canvas(OUT, pagesize=(W, H))
c.setTitle("SIH 2026 - PS 26153 - ARGUS - 6-slide idea submission")
c.setAuthor("SIH Team")

# ---------------------------------------------------------------- SLIDE 1
draw_in_box(c, "SMART INDIA HACKATHON 2026",
            ParagraphStyle("h1", fontName=SERIFB, fontSize=34, leading=40,
                           textColor=NAVY, alignment=PP_ALIGN.CENTER),
            0.4, 0.22, 11.0, 0.75, anchor="middle")
wordmark(c, 11.45, 0.18, 1.75)
tricolor_dots(c, 0.82)
draw_in_box(c, "TITLE PAGE",
            ParagraphStyle("tp", fontName=SERIFB, fontSize=32, leading=38,
                           textColor=BLACK, alignment=PP_ALIGN.CENTER),
            1.0, 1.12, 11.3, 0.8, anchor="middle")

pointers = [
    ("Problem Statement ID – ", "26153", 18),
    ("Problem Statement Title – ",
     "AI-based Network Attack Forecasting from Network Traffic Data using World Models", 16),
    ("Theme – ", "<exact theme from SIH portal>", 18),
    ("PS Category – ", "Software", 18),
    ("Team ID – ", "<from portal>", 18),
    ("Team Name (Registered on portal) – ", "<registered team name>", 18),
]
items = []
for label, val, size in pointers:
    html = (f"<b>{esc(label)}</b>"
            f'<b><font color="#1F4E79">{esc(val)}</font></b>')
    items.append((html, st(f"ptr{size}", SANSB, size, True, BLACK,
                           space_after=11, leading=size * 1.25)))
draw_stack(c, items, 0.65, 2.30, 6.95, h=4.6)

# hook box (radius matches PPTX rounded-rect default: 0.16667 × min side)
rrect(c, 7.95, 2.55, 4.95, 2.15, fill=LBLUE, stroke=NAVY, lw=1.5, radius=0.358)
draw_stack(c, [
    ("ARGUS — an offline AI that predicts the next attack stage "
     "BEFORE the hacker lands, and explains why.",
     st("hook", SANSB, 16.5, True, NAVY, PP_ALIGN.CENTER,
        space_after=6, leading=20)),
    ("Autonomous Attack Forecasting &amp; Multi-Agent SOC Co-Pilot",
     st("hook2", OBL, 12, False, DBLUE, PP_ALIGN.CENTER)),
], 7.95, 2.55, 4.95, 2.15, anchor="middle", margin_x=0.18)

draw_in_box(c, "CORE DOMAIN SKILLS",
            st("cs", SANSB, 9.5, True, GREY), 7.95, 4.74, 4.95, 0.25)
cx = 7.95
for t, cw in [("PyTorch", 0.72), ("FastAPI", 0.72), ("Docker", 0.72),
              ("ATT&amp;CK".replace("&amp;", "&"), 0.72), ("SHAP", 0.72),
              ("Open Source", 1.15)]:
    rrect(c, cx, 5.00, cw, 0.42, fill=NAVY, stroke=NAVY, radius=0.21)
    draw_in_box(c, esc(t),
                st("chp", SANSB, 9.5, True, white, PP_ALIGN.CENTER),
                cx, 5.00, cw, 0.42, anchor="middle", margin_x=0.02)
    cx += cw + 0.07
c.showPage()

# ---------------------------------------------------------------- SLIDE 2
header(c, "IDEA TITLE", page_no=2)
draw_in_box(c, "<u>❖ Proposed Solution (Describe your Idea/Solution/Prototype)</u>",
            ParagraphStyle("sub", fontName=SANSB, fontSize=23, leading=28,
                           textColor=DBLUE),
            0.4, 1.05, 12.5, 0.62, anchor="middle")
draw_in_box(c, "ARGUS — Autonomous Attack Forecasting &amp; Multi-Agent SOC Co-Pilot",
            st("prod", SANSB, 14, True, RED), 0.4, 1.64, 12.5, 0.32)

pipe = [
    ("① INPUT", "Flow CSV / PCAP\ntelemetry", LBLUE, DBLUE),
    ("② WORLD MODEL", "Learns P(S(t+1) | S(t))\n— how the network evolves",
     LORNG, ORANGE),
    ("③ FORECAST", "K-step rollout → attacker's\nnext move + P(infiltration)",
     LYELL, RED),
    ("④ OUTPUT", "MITRE stage + SHAP reasons\n+ offline dashboard",
     LGREEN, GREEN),
]
bx, bw, bgap, by, bh = 0.42, 2.86, 0.42, 2.02, 1.42
for i, (t, d, fill, edge) in enumerate(pipe):
    x = bx + i * (bw + bgap)
    rrect(c, x, by, bw, bh, fill=fill, stroke=edge, lw=1.5, radius=0.237)
    draw_stack(c, [
        (esc(t), st("pt", SANSB, 13, True, edge, PP_ALIGN.CENTER, space_after=4)),
        (esc(d).replace("\n", "<br/>"),
         st("pd", SANS, 10.5, False, BLACK, PP_ALIGN.CENTER)),
    ], x, by, bw, bh, anchor="middle", margin_x=0.06)
    if i < 3:
        ax, ay = x + bw + 0.03, by + bh / 2 - 0.14
        aw, ah = bgap - 0.06, 0.28
        # polygon() takes inch points with y measured from the BOTTOM
        def P(px, py, _H=H):
            return (px, _H / 72 - py)
        polygon(c, [
            P(ax, ay + 0.3 * ah), P(ax + 0.65 * aw, ay + 0.3 * ah),
            P(ax + 0.65 * aw, ay), P(ax + aw, ay + 0.5 * ah),
            P(ax + 0.65 * aw, ay + ah), P(ax + 0.65 * aw, ay + 0.7 * ah),
            P(ax, ay + 0.7 * ah),
        ], NAVY)

bullets_block(c, 0.5, 3.72, 6.15, 3.3, [
    ("✔ Detailed explanation", DBLUE, [
        "Watches traffic as a time-story, not isolated packets",
        "Learns network world model: P(next state | current state)",
        "Forecasts K steps ahead → attack probability before compromise",
        "Maps forecast to MITRE ATT&CK kill-chain stages",
        "Explains every alert: SHAP + attention weights",
        "Runs 100% offline — no cloud APIs, air-gap ready",
    ]),
], size=12.5)
bullets_block(c, 6.85, 3.72, 6.1, 3.3, [
    ("✔ How it addresses the problem", DBLUE, [
        "Today's IDS alarms AFTER entry — 241 days avg to contain (IBM 2025)",
        "ARGUS warns during recon / lateral movement — before exfiltration",
    ]),
    ("✔ Innovation & uniqueness", DBLUE, [
        "1st open-source combo: real flow+packet telemetry → learned "
        "state transitions → K-step attack forecast",
        "Dual engine: Temporal Transformer + zero-dependency Markov fallback",
        "Auto-generates CERT-In (6-hr SLA) & NCIIPC CII advisories",
    ]),
], size=12.5)
c.showPage()

# ---------------------------------------------------------------- SLIDE 3
header(c, "TECHNICAL APPROACH", page_no=3)
strip(c, 0.45, 1.02, 12.4,
      "◆ Technologies to be used   ·   ◆ Methodology & process (flow)   ·   "
      "◆ Working prototype")

cards_l = [
    ("Languages & Core", "Python 3.10 · pandas · Scapy / PyShark", DBLUE),
    ("ML / AI Engine", "PyTorch Temporal Transformer (4-layer, 612K params) · "
     "Random Forest + TreeSHAP · Markov fallback", RED),
    ("Backend & Logic", "FastAPI · SSE streaming · MCP tool server · CLI", DBLUE),
    ("Frontend / UI", "Offline single-page SOC dashboard (HTML/JS) — no internet",
     DBLUE),
]
cards_r = [
    ("Data — 8 loaders", "CIC-IDS-2017/18 · UNSW-NB15 · CTU-13 · CICIoT2023 · "
     "LANL · DARPA/NSL-KDD · synthetic", GREEN),
    ("Knowledge & Standards", "MITRE ATT&CK v14 (697 techniques) · CAPEC · "
     "NIST NVD · NetFlow / PCAP / IPFIX", ORANGE),
    ("Deploy / Hardware", "Docker · CPU-only (no GPU) · fully air-gapped capable",
     GREEN),
    ("✓ Working Prototype", "5-tab dashboard · CLI skills · 37/37 tests green · "
     "1-command docker run", RED),
]

def card(x, y, w, label, value, color):
    rrect(c, x, y, w, 0.74, fill=white, stroke=color, lw=1.5, radius=0.123)
    html = f'<b><font color="{color.hexval().replace("0x", "#")}">{esc(label)}:  </font></b>{esc(value)}'
    draw_in_box(c, html, st("cd", SANS, 10.5, False, BLACK, leading=13),
                x, y, w, 0.74, anchor="middle", margin_x=0.1)

y0, dy, cw2 = 1.40, 0.86, 6.15
for i, (l, v, col) in enumerate(cards_l):
    card(0.45, y0 + i * dy, cw2, l, v, col)
for i, (l, v, col) in enumerate(cards_r):
    card(6.95, y0 + i * dy, cw2, l, v, col)

strip(c, 0.45, 4.92, 12.4, "METHODOLOGY — INPUT → PROCESS → OUTPUT FLOW")
stages = [
    "PCAP /\nFlow CSV",
    "38-feature\nwindows\n(5–10 s)",
    "Sequences\n[t-2, t-1, t0]\n→ t+1",
    "Dual engine\nRF + Temporal\nTransformer",
    "3 heads:\nstate / stage /\nP(infiltration)",
    "K-step\nautoregressive\nrollout",
    "Explain:\nSHAP +\nattention",
    "Dashboard +\nCERT-In /\nNCIIPC report",
]
cx0, cw3, step, cy, chh = 0.32, 1.68, 1.575, 5.28, 1.20
notch = 0.208 * min(cw3, chh)  # match PPTX chevron adjustment 0.208

def chevron(c, x, y, w, h, notch, col):
    """Chevron in inch coords, top-left origin."""
    pts = []
    def P(px, py):
        return (px, H / 72 - py)
    pts = [P(x, y), P(x + w - notch, y), P(x + w, y + h / 2),
           P(x + w - notch, y + h), P(x, y + h), P(x + notch, y + h / 2)]
    polygon(c, pts, col)

for i, t in enumerate(stages):
    x = cx0 + i * step
    col = NAVY if i % 2 == 0 else BLUE
    chevron(c, x, cy, cw3, chh, notch, col)
    draw_in_box(c, esc(t).replace("\n", "<br/>"),
                st("cv", SANSB, 8.5, True, white, PP_ALIGN.CENTER, leading=10.5),
                x + notch + 0.04, cy, cw3 - 2 * notch - 0.08, chh,
                anchor="middle", margin_x=0.02)
draw_in_box(c, "Temporal split — no future leakage  ·  Combined loss: "
            "MSE(state) + CE(stage) + BCE(infiltration) = 0.3 / 0.4 / 0.3",
            ParagraphStyle("fn", fontName=OBL, fontSize=10.5, leading=13,
                           textColor=GREY, alignment=PP_ALIGN.CENTER),
            0.45, 6.60, 12.4, 0.35)
c.showPage()

# ---------------------------------------------------------------- SLIDE 4
header(c, "FEASIBILITY AND VIABILITY", page_no=4)
strip(c, 0.4, 1.02, 12.5,
      "◆ Analysis of feasibility   ·   ◆ Potential challenges and risks   ·   "
      "◆ Strategies for overcoming these challenges")

col_box(c, 0.40, 1.40, 4.00, 4.45, "ANALYSIS OF FEASIBILITY", GREEN, LGREEN, [
    "Technical: 612K-param model trains in ~58 s on CPU; real-time inference",
    "Operational: drop-in on existing NetFlow/PCAP sensors; Docker = 1 command",
    "Economic: 100% open-source → zero cloud / API / licence cost",
    "Data: 8 public benchmark datasets already bundled & normalized",
    "Already prototyped: live dashboard + CLI + 37/37 tests green",
])
col_box(c, 4.66, 1.40, 4.00, 4.45, "POTENTIAL CHALLENGES & RISKS", ORANGE,
        LORNG, [
            "False alarms / alert fatigue on never-seen attack variants",
            "Raw PCAP volume (GBs) & class imbalance in labels",
            "Model drift as attackers change tactics (TTPs)",
            "AI co-pilot itself: prompt-injection & PII-leakage risk",
            "Legacy / air-gapped CII networks with no GPU",
        ])
col_box(c, 8.93, 1.40, 4.00, 4.45, "STRATEGIES TO OVERCOME", BLUE, LBLUE, [
    "Temporal no-leak splits + 8-dataset eval vs LR/RF baseline (FPR per class)",
    "Flow aggregation before inference — never feed raw PCAP to the model",
    "Continuous retraining triggers + kill-chain transition monitoring",
    "Guardrails: PII redaction · RBAC · injection scan · HMAC audit · "
    "human approval",
    "Markov fallback — zero ML dependencies on old hardware",
])

rrect(c, 0.40, 6.05, 12.53, 0.90, fill=NAVY, stroke=NAVY, radius=0.144)
draw_stack(c, [
    ("SECURITY &amp; COMPLIANCE",
     st("bn1", SANSB, 12, True, GOLD, PP_ALIGN.CENTER, space_after=3)),
    ("Air-gapped / no cloud APIs  ·  AES-256 at rest  ·  tamper-evident "
     "SHA-256 HMAC audit  ·  CERT-In 6-hour SLA (Sec 70B)  ·  "
     "NCIIPC CII advisory (Sec 70A — 6 sectors)",
     st("bn2", SANSB, 11.5, True, white, PP_ALIGN.CENTER)),
], 0.40, 6.05, 12.53, 0.90, anchor="middle", margin_x=0.1)
c.showPage()

# ---------------------------------------------------------------- SLIDE 5
header(c, "IMPACT AND BENEFITS", page_no=5)
strip(c, 0.4, 1.02, 12.5,
      "◆ Measurable impact (benchmarked)   ·   ◆ Potential impact on the "
      "target audience   ·   ◆ Benefits (social, economic, environmental)")

metrics = [
    ("99.97%", "World-model accuracy\n(vs 93.0% RF baseline)"),
    ("0.983", "Infiltration probability\nAUC-ROC"),
    ("K = 5", "windows of early warning\nbefore compromise"),
    ("8", "public benchmark\ndatasets supported"),
    ("37/37", "automated tests\ngreen"),
]
mw, mgap, mx, my, mh = 2.36, 0.18, 0.40, 1.42, 1.34
for i, (num, cap) in enumerate(metrics):
    x = mx + i * (mw + mgap)
    rrect(c, x, my, mw, mh, fill=LNAVY, stroke=NAVY, lw=1.5, radius=0.223)
    draw_stack(c, [
        (esc(num), st("m1", SANSB, 25, True, BLUE, PP_ALIGN.CENTER,
                      space_after=3, leading=29)),
        (esc(cap).replace("\n", "<br/>"),
         st("m2", SANS, 9.5, False, GREY, PP_ALIGN.CENTER, leading=12)),
    ], x, my, mw, mh, anchor="middle", margin_x=0.05)

col_box(c, 0.40, 3.05, 6.18, 3.55,
        "POTENTIAL IMPACT ON THE TARGET AUDIENCE", DBLUE, white, [
            "NTRO / NCIIPC: pre-breach forecasting for 6 CII sectors — power, "
            "BFSI, telecom, transport, govt, defence",
            "SOC analysts: a few ranked, explained alerts — not thousands of "
            "noisy late alarms",
            "Enterprises / ISPs / hospitals / railways: affordable, offline, "
            "Indian-made defence stack",
            "Students & open-source community: extensible MIT-licensed base "
            "for Indian cyber research",
        ], bullet_size=12.5)
col_box(c, 6.75, 3.05, 6.18, 3.55,
        "BENEFITS OF THE SOLUTION (SOCIAL · ECONOMIC · STRATEGIC)", DBLUE,
        white, [
            "Social: protects hospital, power-grid & citizen data → trust in "
            "Digital India",
            "Economic: breach avg $4.44M & 241-day containment (IBM 2025) — "
            "ARGUS shortens detect→contain",
            "National / strategic: Atmanirbhar sovereign offline stack — no "
            "foreign cloud, telemetry never leaves",
            "Environmental: CPU-only inference, no GPU farm · Ecosystem: "
            "open-source SOC tooling for India",
        ], bullet_size=12.5)
c.showPage()

# ---------------------------------------------------------------- SLIDE 6
header(c, "RESEARCH AND REFERENCES", page_no=6)
strip(c, 0.4, 1.02, 12.5, "◆ Details / Links of the reference and research work")

col_box(c, 0.40, 1.40, 5.05, 5.35,
        "PROBLEM STATEMENT & STATUTORY REFERENCES", DBLUE, white, [
            "SIH PS 26153 — AI-based Network Attack Forecasting using World "
            "Models (NTRO / NCIIPC), SIH 2026",
            "CERT-In Cyber Security Directions 2022 — 6-hour incident "
            "reporting — https://www.cert-in.org.in",
            "NCIIPC — Sec 70A, IT Act 2000 — CII protection — "
            "https://nciipc.gov.in",
            "MITRE ATT&CK v14 — https://attack.mitre.org  ·  CAPEC  ·  "
            "NIST NVD API 2.0 (CVSS v3.1)",
            "IBM Cost of a Data Breach Report 2025 — $4.44M avg · 241-day "
            "containment — https://ibm.com/reports/data-breach",
            "Datasets: CIC-IDS-2017/18 — "
            "https://www.unb.ca/cic/datasets/ids-2017.html",
        ], bullet_size=11.5, bullet_space=7)
col_box(c, 5.72, 1.40, 7.21, 5.35, "RESEARCH FOUNDATIONS (RELATED WORK)",
        DBLUE, white, [
            "Field survey — Husák, Komárková, Bou-Harb, Čeleda: “Survey of "
            "Attack Projection, Prediction and Forecasting in Cyber Security”, "
            "IEEE COMST 2019",
            "World models — Ha & Schmidhuber 2018, arxiv.org/abs/1803.10122 · "
            "DreamerV3, Hafner et al. 2023",
            "Security sequence prediction — Tiresias (CCS 2018) · DeepLog "
            "(CCS 2017) · DeepCASE (IEEE S&P 2022)",
            "Graph / provenance NIDS — E-GraphSAGE · Anomal-E · Kairos "
            "(IEEE S&P 2024)",
            "Explainability — SHAP: Lundberg & Lee, arxiv.org/abs/1705.07874  ·  "
            "attention-weight attribution",
            "More datasets — UNSW-NB15 · CTU-13 · CICIoT2023 · LANL auth "
            "(loaders in ml/world_model/dataset_loader.py)",
            "Reproducible baseline harness — World Model vs Logistic "
            "Regression: ml/world_model/benchmark.py",
        ], bullet_size=11.5, bullet_space=7)
c.showPage()

c.save()
print("saved:", OUT)

# ------------------------------------------------------- previews + checks
import pymupdf  # noqa: E402
os.makedirs(PNG_DIR, exist_ok=True)
doc = pymupdf.open(OUT)
for i, page in enumerate(doc, 1):
    pix = page.get_pixmap(dpi=110)
    pix.save(f"{PNG_DIR}/slide{i}.png")
print("previews:", PNG_DIR, f"({len(doc)} pages)")
