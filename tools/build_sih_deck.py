#!/usr/bin/env python3
"""Build the SIH 2026 6-slide template-compliant deck for PS 26153 (ARGUS).

Mirrors the official SIH idea-submission template:
  1 TITLE PAGE · 2 IDEA TITLE · 3 TECHNICAL APPROACH
  4 FEASIBILITY AND VIABILITY · 5 IMPACT AND BENEFITS · 6 RESEARCH AND REFERENCES
Footer band + page numbers on slides 2-6, SIH wordmark top-right on all slides.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---------------------------------------------------------------- palette
NAVY   = RGBColor(0x1F, 0x38, 0x64)
BLUE   = RGBColor(0x0E, 0x76, 0xBE)
DBLUE  = RGBColor(0x1F, 0x4E, 0x79)
BLACK  = RGBColor(0x1A, 0x1A, 0x1A)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
GREY   = RGBColor(0x44, 0x44, 0x44)
LGREY  = RGBColor(0x88, 0x88, 0x88)
PURPLE = RGBColor(0x7C, 0x4D, 0xA0)
GREEN  = RGBColor(0x1E, 0x8E, 0x3E)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
RED    = RGBColor(0xC0, 0x39, 0x2B)
LBLUE  = RGBColor(0xEA, 0xF2, 0xFB)
LGREEN = RGBColor(0xEC, 0xF7, 0xEF)
LORNG  = RGBColor(0xFD, 0xF2, 0xE9)
LYELL  = RGBColor(0xFD, 0xF6, 0xE3)
LNAVY  = RGBColor(0xEA, 0xF0, 0xF8)
SAFF   = RGBColor(0xFF, 0x99, 0x33)
INKGRN = RGBColor(0x13, 0x88, 0x08)

SERIF = "Times New Roman"
SANS  = "Arial"

SW, SH = 13.333, 7.5
FOOT_Y, FOOT_H = 7.15, 0.35

prs = Presentation()
prs.slide_width  = Inches(SW)
prs.slide_height = Inches(SH)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- helpers
def txbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP, wrap=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    return tb, tf


def set_run(run, text, size, bold=False, color=BLACK, font=SANS,
            italic=False, underline=False):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.underline = underline
    run.font.color.rgb = color
    run.font.name = font


def para(tf, first=False, align=PP_ALIGN.LEFT, space_before=0, space_after=4,
         line=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    return p


def add_rrect(slide, x, y, w, h, fill=None, line=None, line_w=1.0,
              shape=MSO_SHAPE.ROUNDED_RECTANGLE, adj=None):
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(line_w)
    sp.shadow.inherit = False
    if adj is not None and sp.adjustments:
        sp.adjustments[0] = adj
    return sp


def fill_shape(sp, blocks, anchor=MSO_ANCHOR.MIDDLE, margin=0.06):
    """Fill an autoshape's text frame.

    blocks = [(text, size, bold, color, align, space_after), ...]
    Newlines in text become separate paragraphs (runs don't render \\n).
    """
    tf = sp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    first = True
    for text, size, bold, color, align, sa in blocks:
        for line_text in text.split("\n"):
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.alignment = align
            p.space_after = Pt(sa)
            r = p.add_run()
            set_run(r, line_text, size, bold=bold, color=color)
    return sp


def header(slide, title, page_no=None, team=True):
    """SIH content-slide chrome: title, team bubble, wordmark, footer, page #."""
    # centered serif title
    _, tf = txbox(slide, 1.75, 0.18, 9.8, 0.85, anchor=MSO_ANCHOR.MIDDLE)
    p = para(tf, first=True, align=PP_ALIGN.CENTER, space_after=0)
    set_run(p.add_run(), title, 33, bold=True, color=BLACK, font=SERIF)
    # team bubble (template: purple ellipse, top-left)
    if team:
        ov = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.25), Inches(0.12),
                                    Inches(1.55), Inches(0.8))
        ov.fill.background()
        ov.line.color.rgb = PURPLE
        ov.line.width = Pt(1.5)
        ov.shadow.inherit = False
        fill_shape(ov, [("Your\nTeam Name", 11, False, BLACK, PP_ALIGN.CENTER, 0)])
    # SIH wordmark stand-in, top-right (replace with official logo image)
    _, tf = txbox(slide, 11.35, 0.14, 1.85, 0.8)
    p = para(tf, first=True, align=PP_ALIGN.RIGHT, space_after=0)
    set_run(p.add_run(), "SMART INDIA", 8.5, bold=True, color=NAVY, font=SANS)
    p = para(tf, align=PP_ALIGN.RIGHT, space_after=0)
    set_run(p.add_run(), "HACKATHON", 8.5, bold=True, color=NAVY, font=SANS)
    p = para(tf, align=PP_ALIGN.RIGHT, space_after=0)
    set_run(p.add_run(), "2026", 12, bold=True, color=INKGRN, font=SANS)
    # tricolor dots
    for i, c in enumerate((SAFF, WHITE, INKGRN)):
        d = slide.shapes.add_shape(MSO_SHAPE.OVAL,
                                   Inches(12.55 + i * 0.16), Inches(0.78),
                                   Inches(0.13), Inches(0.13))
        d.fill.solid(); d.fill.fore_color.rgb = c
        d.line.color.rgb = GREY; d.line.width = Pt(0.5)
        d.shadow.inherit = False
    # footer band
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0),
                                 Inches(FOOT_Y), Inches(SW), Inches(FOOT_H))
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background(); bar.shadow.inherit = False
    _, tf = txbox(slide, 0, FOOT_Y + 0.04, SW, 0.28)
    p = para(tf, first=True, align=PP_ALIGN.CENTER, space_after=0)
    set_run(p.add_run(), "@SIH Idea submission- Template", 11, color=WHITE)
    if page_no is not None:
        _, tf = txbox(slide, SW - 0.75, FOOT_Y + 0.04, 0.5, 0.28)
        p = para(tf, first=True, align=PP_ALIGN.RIGHT, space_after=0)
        set_run(p.add_run(), str(page_no), 11, bold=True, color=WHITE)


def col_box(slide, x, y, w, h, title, tcolor, body_fill, bullets,
            title_size=13, bullet_size=11.5, bullet_space=5):
    """Colored header bar + bordered body with bullets."""
    hb = add_rrect(slide, x, y, w, 0.44, fill=tcolor, line=tcolor, adj=0.18)
    fill_shape(hb, [(title, title_size, True, WHITE, PP_ALIGN.CENTER, 0)])
    body = add_rrect(slide, x, y + 0.40, w, h - 0.40, fill=body_fill,
                     line=tcolor, line_w=1.25)
    tf = body.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.08)
    first = True
    for b in bullets:
        p = para(tf, first=first, space_after=bullet_space)
        first = False
        set_run(p.add_run(), "• " + b, bullet_size, color=BLACK)
    return body


def bullets_block(slide, x, y, w, h, groups, size=12):
    """groups = [(label, color, [bullet, ...]), ...]"""
    _, tf = txbox(slide, x, y, w, h)
    first = True
    for label, lcolor, items in groups:
        p = para(tf, first=first, space_before=0 if first else 7, space_after=3)
        first = False
        set_run(p.add_run(), label, size + 1, bold=True, color=lcolor)
        for b in items:
            p = para(tf, space_after=3)
            set_run(p.add_run(), "• " + b, size, color=BLACK)
    return tf


# ================================================================ SLIDE 1
s = prs.slides.add_slide(BLANK)
# top banner title (template style)
_, tf = txbox(s, 0.4, 0.22, 11.0, 0.75, anchor=MSO_ANCHOR.MIDDLE)
p = para(tf, first=True, align=PP_ALIGN.CENTER, space_after=0)
set_run(p.add_run(), "SMART INDIA HACKATHON 2026", 34, bold=True,
        color=NAVY, font=SERIF)
# wordmark top-right (same as content slides)
_, tf = txbox(s, 11.45, 0.18, 1.75, 0.8)
p = para(tf, first=True, align=PP_ALIGN.RIGHT, space_after=0)
set_run(p.add_run(), "SMART INDIA", 8.5, bold=True, color=NAVY)
p = para(tf, align=PP_ALIGN.RIGHT, space_after=0)
set_run(p.add_run(), "HACKATHON", 8.5, bold=True, color=NAVY)
p = para(tf, align=PP_ALIGN.RIGHT, space_after=0)
set_run(p.add_run(), "2026", 12, bold=True, color=INKGRN)
for i, c in enumerate((SAFF, WHITE, INKGRN)):
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(12.55 + i * 0.16),
                           Inches(0.82), Inches(0.13), Inches(0.13))
    d.fill.solid(); d.fill.fore_color.rgb = c
    d.line.color.rgb = GREY; d.line.width = Pt(0.5); d.shadow.inherit = False
# PAGE heading
_, tf = txbox(s, 1.0, 1.12, 11.3, 0.8, anchor=MSO_ANCHOR.MIDDLE)
p = para(tf, first=True, align=PP_ALIGN.CENTER, space_after=0)
set_run(p.add_run(), "TITLE PAGE", 32, bold=True, color=BLACK, font=SERIF)

# left: template pointers
pointers = [
    ("Problem Statement ID – ", "26153", 18),
    ("Problem Statement Title – ", "AI-based Network Attack Forecasting from "
     "Network Traffic Data using World Models", 16),
    ("Theme – ", "<exact theme from SIH portal>", 18),
    ("PS Category – ", "Software", 18),
    ("Team ID – ", "<from portal>", 18),
    ("Team Name (Registered on portal) – ", "<registered team name>", 18),
]
_, tf = txbox(s, 0.65, 2.30, 6.95, 4.6)
first = True
for label, val, size in pointers:
    p = para(tf, first=first, space_after=11)
    first = False
    set_run(p.add_run(), label, size, bold=True, color=BLACK)
    set_run(p.add_run(), val, size, bold=True, color=DBLUE)

# right: hook box + skills chips
hb = add_rrect(s, 7.95, 2.55, 4.95, 2.15, fill=LBLUE, line=NAVY, line_w=1.5)
tfh = hb.text_frame; tfh.word_wrap = True
tfh.vertical_anchor = MSO_ANCHOR.MIDDLE
tfh.margin_left = tfh.margin_right = Inches(0.18)
p = para(tfh, first=True, align=PP_ALIGN.CENTER, space_after=6)
set_run(p.add_run(),
        "ARGUS — an offline AI that predicts the next attack stage "
        "BEFORE the hacker lands, and explains why.", 16.5, bold=True,
        color=NAVY, font=SANS)
p = para(tfh, align=PP_ALIGN.CENTER, space_after=0)
set_run(p.add_run(), "Autonomous Attack Forecasting & Multi-Agent SOC Co-Pilot",
        12, italic=True, color=DBLUE)
# skills chips (widths tuned to stay inside the 7.95–12.90 column)
chips = [("PyTorch", 0.72), ("FastAPI", 0.72), ("Docker", 0.72),
         ("ATT&CK", 0.72), ("SHAP", 0.72), ("Open Source", 1.15)]
cx, gap = 7.95, 0.07
for t, w in chips:
    ch = add_rrect(s, cx, 5.00, w, 0.42, fill=NAVY, line=NAVY, adj=0.5)
    fill_shape(ch, [(t, 9.5, True, WHITE, PP_ALIGN.CENTER, 0)], margin=0.02)
    cx += w + gap
# skills label
_, tf = txbox(s, 7.95, 4.74, 4.95, 0.25)
p = para(tf, first=True, space_after=0)
set_run(p.add_run(), "CORE DOMAIN SKILLS", 9.5, bold=True, color=GREY)

# ================================================================ SLIDE 2
s = prs.slides.add_slide(BLANK)
header(s, "IDEA TITLE", page_no=2)
# template pointer line (blue, underlined)
_, tf = txbox(s, 0.4, 1.05, 12.5, 0.62, anchor=MSO_ANCHOR.MIDDLE)
p = para(tf, first=True, space_after=0)
set_run(p.add_run(), "❖ Proposed Solution (Describe your Idea/Solution/Prototype)",
        23, bold=True, color=DBLUE, underline=True)
_, tf = txbox(s, 0.4, 1.64, 12.5, 0.32)
p = para(tf, first=True, space_after=0)
set_run(p.add_run(),
        "ARGUS — Autonomous Attack Forecasting & Multi-Agent SOC Co-Pilot",
        14, bold=True, color=RED)

# pipeline: 4 boxes + arrows
pipe = [
    ("① INPUT", "Flow CSV / PCAP\ntelemetry", LBLUE, DBLUE),
    ("② WORLD MODEL", "Learns P(S(t+1) | S(t))\n— how the network evolves", LORNG, ORANGE),
    ("③ FORECAST", "K-step rollout → attacker's\nnext move + P(infiltration)", LYELL, RED),
    ("④ OUTPUT", "MITRE stage + SHAP reasons\n+ offline dashboard", LGREEN, GREEN),
]
bx, bw, bgap, by, bh = 0.42, 2.86, 0.42, 2.02, 1.42
for i, (t, d, fill, edge) in enumerate(pipe):
    x = bx + i * (bw + bgap)
    sp = add_rrect(s, x, by, bw, bh, fill=fill, line=edge, line_w=1.5)
    fill_shape(sp, [
        (t, 13, True, edge, PP_ALIGN.CENTER, 4),
        (d, 10.5, False, BLACK, PP_ALIGN.CENTER, 0),
    ], margin=0.06)
    if i < 3:
        ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                Inches(x + bw + 0.03), Inches(by + bh / 2 - 0.14),
                                Inches(bgap - 0.06), Inches(0.28))
        ar.fill.solid(); ar.fill.fore_color.rgb = NAVY
        ar.line.fill.background(); ar.shadow.inherit = False

# three pointer groups, two columns
bullets_block(s, 0.5, 3.72, 6.15, 3.3, [
    ("✔ Detailed explanation", DBLUE, [
        "Watches traffic as a time-story, not isolated packets",
        "Learns network world model: P(next state | current state)",
        "Forecasts K steps ahead → attack probability before compromise",
        "Maps forecast to MITRE ATT&CK kill-chain stages",
        "Explains every alert: SHAP + attention weights",
        "Runs 100% offline — no cloud APIs, air-gap ready",
    ]),
], size=12.5)
bullets_block(s, 6.85, 3.72, 6.1, 3.3, [
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

# ================================================================ SLIDE 3
s = prs.slides.add_slide(BLANK)
header(s, "TECHNICAL APPROACH", page_no=3)
_, tf = txbox(s, 0.45, 1.02, 12.4, 0.3)
p = para(tf, first=True, space_after=0)
set_run(p.add_run(),
        "◆ Technologies to be used   ·   ◆ Methodology & process (flow)   ·   "
        "◆ Working prototype", 12, bold=True, color=GREY)

cards_l = [
    ("Languages & Core", "Python 3.10 · pandas · Scapy / PyShark", DBLUE),
    ("ML / AI Engine", "PyTorch Temporal Transformer (4-layer, 612K params) · "
     "Random Forest + TreeSHAP · Markov fallback", RED),
    ("Backend & Logic", "FastAPI · SSE streaming · MCP tool server · CLI", DBLUE),
    ("Frontend / UI", "Offline single-page SOC dashboard (HTML/JS) — no internet", DBLUE),
]
cards_r = [
    ("Data — 8 loaders", "CIC-IDS-2017/18 · UNSW-NB15 · CTU-13 · CICIoT2023 · "
     "LANL · DARPA/NSL-KDD · synthetic", GREEN),
    ("Knowledge & Standards", "MITRE ATT&CK v14 (697 techniques) · CAPEC · "
     "NIST NVD · NetFlow / PCAP / IPFIX", ORANGE),
    ("Deploy / Hardware", "Docker · CPU-only (no GPU) · fully air-gapped capable", GREEN),
    ("✓ Working Prototype", "5-tab dashboard · CLI skills · 37/37 tests green · "
     "1-command docker run", RED),
]
def card_row(x, y, w, label, value, color):
    sp = add_rrect(s, x, y, w, 0.74, fill=WHITE, line=color, line_w=1.5)
    tf = sp.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.1)
    p = para(tf, first=True, space_after=0)
    set_run(p.add_run(), label + ":  ", 11, bold=True, color=color)
    set_run(p.add_run(), value, 10.5, color=BLACK)

y0, dy, cw2 = 1.40, 0.86, 6.15
for i, (l, v, c) in enumerate(cards_l):
    card_row(0.45, y0 + i * dy, cw2, l, v, c)
for i, (l, v, c) in enumerate(cards_r):
    card_row(6.95, y0 + i * dy, cw2, l, v, c)

# methodology chevron flow
_, tf = txbox(s, 0.45, 4.92, 12.4, 0.3)
p = para(tf, first=True, space_after=0)
set_run(p.add_run(), "METHODOLOGY — INPUT → PROCESS → OUTPUT FLOW",
        12, bold=True, color=GREY)
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
cx, cw3, step, cy, chh = 0.32, 1.68, 1.575, 5.28, 1.20
for i, t in enumerate(stages):
    x = cx + i * step
    ch = s.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x), Inches(cy),
                            Inches(cw3), Inches(chh))
    ch.adjustments[0] = 0.208  # smaller notch so white text stays on the fill
    ch.fill.solid()
    ch.fill.fore_color.rgb = NAVY if i % 2 == 0 else BLUE
    ch.line.fill.background(); ch.shadow.inherit = False
    fill_shape(ch, [(t, 9, True, WHITE, PP_ALIGN.CENTER, 0)], margin=0.03)
_, tf = txbox(s, 0.45, 6.60, 12.4, 0.35)
p = para(tf, first=True, align=PP_ALIGN.CENTER, space_after=0)
set_run(p.add_run(),
        "Temporal split — no future leakage  ·  Combined loss: "
        "MSE(state) + CE(stage) + BCE(infiltration) = 0.3 / 0.4 / 0.3",
        10.5, italic=True, color=GREY)

# ================================================================ SLIDE 4
s = prs.slides.add_slide(BLANK)
header(s, "FEASIBILITY AND VIABILITY", page_no=4)
_, tf = txbox(s, 0.4, 1.02, 12.5, 0.3)
p = para(tf, first=True, space_after=0)
set_run(p.add_run(),
        "◆ Analysis of feasibility   ·   ◆ Potential challenges and risks   ·   "
        "◆ Strategies for overcoming these challenges", 12, bold=True, color=GREY)

col_box(s, 0.40, 1.40, 4.00, 4.45,
        "ANALYSIS OF FEASIBILITY", GREEN, LGREEN, [
            "Technical: 612K-param model trains in ~58 s on CPU; real-time inference",
            "Operational: drop-in on existing NetFlow/PCAP sensors; Docker = 1 command",
            "Economic: 100% open-source → zero cloud / API / licence cost",
            "Data: 8 public benchmark datasets already bundled & normalized",
            "Already prototyped: live dashboard + CLI + 37/37 tests green",
        ], bullet_size=11.5)
col_box(s, 4.66, 1.40, 4.00, 4.45,
        "POTENTIAL CHALLENGES & RISKS", ORANGE, LORNG, [
            "False alarms / alert fatigue on never-seen attack variants",
            "Raw PCAP volume (GBs) & class imbalance in labels",
            "Model drift as attackers change tactics (TTPs)",
            "AI co-pilot itself: prompt-injection & PII-leakage risk",
            "Legacy / air-gapped CII networks with no GPU",
        ], bullet_size=11.5)
col_box(s, 8.93, 1.40, 4.00, 4.45,
        "STRATEGIES TO OVERCOME", BLUE, LBLUE, [
            "Temporal no-leak splits + 8-dataset eval vs LR/RF baseline (FPR per class)",
            "Flow aggregation before inference — never feed raw PCAP to the model",
            "Continuous retraining triggers + kill-chain transition monitoring",
            "Guardrails: PII redaction · RBAC · injection scan · HMAC audit · "
            "human approval",
            "Markov fallback — zero ML dependencies on old hardware",
        ], bullet_size=11.5)

# security banner
bn = add_rrect(s, 0.40, 6.05, 12.53, 0.90, fill=NAVY, line=NAVY, adj=0.16)
fill_shape(bn, [
    ("SECURITY & COMPLIANCE", 12, True, RGBColor(0xFF, 0xD5, 0x4F),
     PP_ALIGN.CENTER, 3),
    ("Air-gapped / no cloud APIs  ·  AES-256 at rest  ·  tamper-evident "
     "SHA-256 HMAC audit  ·  CERT-In 6-hour SLA (Sec 70B)  ·  "
     "NCIIPC CII advisory (Sec 70A — 6 sectors)", 11.5, True, WHITE,
     PP_ALIGN.CENTER, 0),
], margin=0.1)

# ================================================================ SLIDE 5
s = prs.slides.add_slide(BLANK)
header(s, "IMPACT AND BENEFITS", page_no=5)
_, tf = txbox(s, 0.4, 1.02, 12.5, 0.3)
p = para(tf, first=True, space_after=0)
set_run(p.add_run(),
        "◆ Measurable impact (benchmarked)   ·   ◆ Potential impact on the "
        "target audience   ·   ◆ Benefits (social, economic, environmental)",
        12, bold=True, color=GREY)

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
    sp = add_rrect(s, x, my, mw, mh, fill=LNAVY, line=NAVY, line_w=1.5)
    fill_shape(sp, [
        (num, 25, True, BLUE, PP_ALIGN.CENTER, 3),
        (cap, 9.5, False, GREY, PP_ALIGN.CENTER, 0),
    ], margin=0.05)

col_box(s, 0.40, 3.05, 6.18, 3.55,
        "POTENTIAL IMPACT ON THE TARGET AUDIENCE", DBLUE, WHITE, [
            "NTRO / NCIIPC: pre-breach forecasting for 6 CII sectors — power, "
            "BFSI, telecom, transport, govt, defence",
            "SOC analysts: a few ranked, explained alerts — not thousands of "
            "noisy late alarms",
            "Enterprises / ISPs / hospitals / railways: affordable, offline, "
            "Indian-made defence stack",
            "Students & open-source community: extensible MIT-licensed base "
            "for Indian cyber research",
        ], bullet_size=12.5)
col_box(s, 6.75, 3.05, 6.18, 3.55,
        "BENEFITS OF THE SOLUTION (SOCIAL · ECONOMIC · STRATEGIC)", DBLUE, WHITE, [
            "Social: protects hospital, power-grid & citizen data → trust in "
            "Digital India",
            "Economic: breach avg $4.44M & 241-day containment (IBM 2025) — "
            "ARGUS shortens detect→contain",
            "National / strategic: Atmanirbhar sovereign offline stack — no "
            "foreign cloud, telemetry never leaves",
            "Environmental: CPU-only inference, no GPU farm · Ecosystem: "
            "open-source SOC tooling for India",
        ], bullet_size=12.5)

# ================================================================ SLIDE 6
s = prs.slides.add_slide(BLANK)
header(s, "RESEARCH AND REFERENCES", page_no=6)
_, tf = txbox(s, 0.4, 1.02, 12.5, 0.3)
p = para(tf, first=True, space_after=0)
set_run(p.add_run(), "◆ Details / Links of the reference and research work",
        12, bold=True, color=GREY)

col_box(s, 0.40, 1.40, 5.05, 5.35,
        "PROBLEM STATEMENT & STATUTORY REFERENCES", DBLUE, WHITE, [
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
            "Datasets: CIC-IDS-2017/18 — https://www.unb.ca/cic/datasets/ids-2017.html",
        ], bullet_size=11.5, bullet_space=7)
col_box(s, 5.72, 1.40, 7.21, 5.35,
        "RESEARCH FOUNDATIONS (RELATED WORK)", DBLUE, WHITE, [
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

# ---------------------------------------------------------------- save
out = "/home/user/Argus/docs/SIH_PS26153_ARGUS_6slide.pptx"
prs.save(out)
print("saved:", out)

# quick self-check
prs2 = Presentation(out)
print("slides:", len(prs2.slides))
for i, sl in enumerate(prs2.slides, 1):
    texts = [sh.text_frame.text for sh in sl.shapes
             if sh.has_text_frame and sh.text_frame.text.strip()]
    chars = sum(len(t) for t in texts)
    print(f"  slide {i}: {len(sl.shapes)} shapes, {chars} text chars")
