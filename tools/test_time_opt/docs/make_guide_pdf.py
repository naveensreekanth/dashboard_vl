"""Generate a compact 2-page PDF user guide for the dashboard."""

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "images"
OUT = ROOT / "Dashboard_User_Guide.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 12 * mm

INK = HexColor("#1a1f26")
MUTED = HexColor("#5a6570")
ACCENT = HexColor("#1f6b4a")
RULE = HexColor("#c8d0d8")
CARD_BG = HexColor("#f4f6f8")


def style_sheet():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=INK,
            alignment=TA_CENTER,
            spaceAfter=2 * mm,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=3 * mm,
        ),
        "h": ParagraphStyle(
            "h",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=ACCENT,
            spaceBefore=2.2 * mm,
            spaceAfter=1.2 * mm,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=1.4 * mm,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.2,
            textColor=MUTED,
            alignment=TA_LEFT,
            spaceAfter=1 * mm,
        ),
        "step": ParagraphStyle(
            "step",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=INK,
            leftIndent=2 * mm,
            spaceAfter=0.6 * mm,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }
    return styles


def fit_image(path: Path, max_w, max_h):
    if not path.exists():
        return None
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def section(styles, title, body, image_name=None, img_w=None, img_h=None):
    bits = [Paragraph(title, styles["h"]), Paragraph(body, styles["body"])]
    if image_name and img_w and img_h:
        img = fit_image(IMG / image_name, img_w, img_h)
        if img:
            bits.append(Spacer(1, 0.8 * mm))
            bits.append(img)
            bits.append(Spacer(1, 1.2 * mm))
    return KeepTogether(bits)


def build():
    styles = style_sheet()
    content_w = PAGE_W - 2 * MARGIN

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=10 * mm,
        bottomMargin=12 * mm,
        title="Verilumen Dashboard User Guide",
        author="Verilumen Simulation Agent",
    )

    story = []

    # ---- Page 1 content ----
    story.append(Paragraph("Test Time Optimization", styles["title"]))
    story.append(
        Paragraph(
            "Verilumen Simulation Agent — Dashboard User Guide (2 pages)",
            styles["subtitle"],
        )
    )

    overview = fit_image(IMG / "00-full-dashboard.png", content_w, 42 * mm)
    if overview:
        story.append(overview)
        story.append(Spacer(1, 1.5 * mm))

    story.append(
        Paragraph(
            "This guide explains the dashboard in simple language. The tool compares a "
            "<b>without verilumen agent</b> (more memory and test time) with "
            "<b>with verilumen agent</b> (fewer patterns, lower memory and faster test).",
            styles["body"],
        )
    )

    story.append(Paragraph("How to use the dashboard", styles["h"]))
    for i, t in enumerate(
        [
            "Upload a STIL file from the left panel.",
            "Click <b>Run live simulation</b>. "
            "<b>This will take some time to load</b> — please wait while processing runs. "
            "The status line and progress bar show embedding and comparison as results load. "
            "Do not move on until the run finishes (status shows <b>Done</b>).",
            "Review the first ten picks, memory gauges, and test-time gauges "
            "(without vs top 10 pick of verilumen agent).",
            "Add fail-log folders from your own computer.",
            "Click <b>Analyze fail counts</b> for <b>high recommended patterns order</b> "
            "and <b>low risk patterns</b>.",
        ],
        start=1,
    ):
        story.append(Paragraph(f"<b>Step {i}.</b> {t}", styles["step"]))

    story.append(Paragraph("1. Title area", styles["h"]))
    story.append(
        Paragraph(
            "The header shows the product purpose: <b>test time optimization</b> with the "
            "Verilumen simulation agent. The yellow flow line "
            "summarizes the ATE path: STIL → compile → vector memory → playback → DUT pins.",
            styles["body"],
        )
    )
    img = fit_image(IMG / "01-header.png", content_w * 0.92, 16 * mm)
    if img:
        story.append(img)
        story.append(Spacer(1, 1 * mm))

    story.append(Paragraph("2. Left panel — STIL and fail logs", styles["h"]))
    story.append(
        Paragraph(
            "Use this panel to start work. <b>Step 1:</b> upload a <b>.stil</b> file. "
            "<b>Step 2:</b> click <b>Run live simulation</b>. "
            "<b>This will take some time to load</b> — wait for processing while the Verilumen agent "
            "embeds patterns and builds the comparison. "
            "Watch the status line and progress bar until the run is <b>Done</b>. "
            "Use <b>Stop</b> only if you need to cancel. "
            "After the run finishes, add log folders with <b>Add log folder</b> "
            "(repeat for more folders) and click <b>Analyze fail counts</b>. "
            "Fail analysis runs locally in the browser — log files stay on your PC.",
            styles["body"],
        )
    )

    # Two small images side by side: sidebar + first10
    left = fit_image(IMG / "02-sidebar.png", content_w * 0.32, 48 * mm)
    right = fit_image(IMG / "04-first10.png", content_w * 0.62, 48 * mm)
    if left and right:
        tbl = Table([[left, right]], colWidths=[content_w * 0.34, content_w * 0.64])
        tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
        story.append(tbl)
        story.append(Spacer(1, 1 * mm))

    story.append(Paragraph("3. First ten patterns (without vs top 10 pick of verilumen agent)", styles["h"]))
    story.append(
        Paragraph(
            "The left board (<b>without verilumen agent order</b>) shows patterns in ordinary order. The right board "
            "(<b>top 10 pick of verilumen agent</b>) shows diverse patterns the agent chose to <b>KEEP</b>. "
            "Click slots or use Prev/Next to replay each decision. Yellow-outlined bits are "
            "different from the already-kept comparison pattern.",
            styles["body"],
        )
    )

    story.append(Paragraph("4. Status and progress", styles["h"]))
    story.append(
        Paragraph(
            "The status line explains the current step (embedding, comparing, or done). "
            "The yellow progress bar fills as the simulation advances.",
            styles["body"],
        )
    )

    # ---- Page 2 forced via spacer calculation - use page break ----
    from reportlab.platypus import PageBreak

    story.append(PageBreak())

    story.append(Paragraph("5. Vector memory and test time results", styles["h"]))
    story.append(
        Paragraph(
            "Two red/green gauge pairs show the main optimization results. "
            "<b>without verilumen agent</b> (red) is the cost of loading/running all patterns. "
            "<b>with verilumen agent</b> (green) is the cost of the kept subset. "
            "Lower green values mean memory and time were saved.",
            styles["body"],
        )
    )

    ram = fit_image(IMG / "05-ram.png", content_w * 0.48, 22 * mm)
    tim = fit_image(IMG / "06-time.png", content_w * 0.48, 22 * mm)
    if ram and tim:
        tbl = Table([[ram, tim]], colWidths=[content_w * 0.5, content_w * 0.5])
        tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
        story.append(tbl)
        story.append(Spacer(1, 1.2 * mm))

    story.append(Paragraph("6. Metrics, chart, and final summary", styles["h"]))
    story.append(
        Paragraph(
            "The metrics row shows Host RSS, pattern count, pins, cycles, and test time saved. "
            "The live chart plots without-verilumen-agent memory (red) against with-verilumen-agent (green). "
            "When the run completes, the green banner reports kept/total patterns, memory, "
            "test time, and percentage savings — the main scoreboard for sharing results.",
            styles["body"],
        )
    )
    summary = fit_image(IMG / "09-summary.png", content_w, 18 * mm)
    if summary:
        story.append(summary)
        story.append(Spacer(1, 1.2 * mm))

    story.append(Paragraph("7. High recommended vs low risk pattern lists", styles["h"]))
    story.append(
        Paragraph(
            "The green list (<b>high recommended patterns order</b>) shows patterns loaded into vector memory. "
            "The red list (<b>low risk patterns</b>) shows patterns skipped to save RAM/time "
            "(usually too similar to an already-kept pattern). "
            "Click a low-risk chip to read why it was not loaded and to see a "
            "<b>full 234-bit comparison</b> against the closest kept pattern "
            "(embedding + bit distance shown; yellow cells = differing 0/1 bits, "
            "same style as the first-ten picker). After fail analysis, badges "
            "like <b>8F</b> mean that pattern had 8 fails in your logs.",
            styles["body"],
        )
    )
    pats = fit_image(IMG / "10-patterns.png", content_w, 26 * mm)
    if pats:
        story.append(pats)
        story.append(Spacer(1, 1.2 * mm))

    high = fit_image(IMG / "10a-high-recommended.png", content_w * 0.48, 48 * mm)
    low = fit_image(IMG / "10b-low-risk.png", content_w * 0.48, 48 * mm)
    cap_high = Paragraph(
        "<b>High recommended patterns order</b> — full pick-order list of kept patterns "
        "(scroll to review every loaded id).",
        styles["small"],
    )
    cap_low = Paragraph(
        "<b>Low risk patterns</b> — skipped patterns; click a chip to see why it was not loaded "
        "(closest kept pattern and distance).",
        styles["small"],
    )
    if high and low:
        tbl = Table(
            [[cap_high, cap_low], [high, low]],
            colWidths=[content_w * 0.5, content_w * 0.5],
        )
        tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 1.5 * mm),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 1.2 * mm))
    else:
        if high:
            story.append(cap_high)
            story.append(high)
            story.append(Spacer(1, 1 * mm))
        if low:
            story.append(cap_low)
            story.append(low)
            story.append(Spacer(1, 1.2 * mm))

    story.append(Paragraph("8. Fail count analysis", styles["h"]))
    story.append(
        Paragraph(
            "After you add logs and click <b>Analyze fail counts</b>, summary cards show "
            "folder count, kept fails, discarded fails, and files parsed. Two tables list "
            "patterns with the most fails — left <b>high recommended patterns fails</b>, "
            "right <b>low risk patterns fail</b> — "
            "sorted highest-first so you can judge coverage risk quickly.",
            styles["body"],
        )
    )
    fails = fit_image(IMG / "11-fails.png", content_w, 36 * mm)
    if fails:
        story.append(fails)
        story.append(Spacer(1, 1.5 * mm))

    story.append(Paragraph("Reading the colors & practical notes", styles["h"]))
    story.append(
        Paragraph(
            "<b>Green</b> = Verilumen / kept / savings. <b>Blue</b> = without agent / low-risk / not loaded. "
            "<b>Yellow</b> = progress and bit differences. Always run the simulation before analyzing "
            "logs. Expected log style uses pattern headers such as P12 and STATUS:F / STATUS:P. "
            "Log labels P1…P1000 map to dashboard ids 0…999 automatically. "
            "Live URL: https://vector-memory-optimization.onrender.com",
            styles["body"],
        )
    )

    def add_page_decor(canvas, doc_):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - 7 * mm, PAGE_W - MARGIN, PAGE_H - 7 * mm)
        canvas.line(MARGIN, 8 * mm, PAGE_W - MARGIN, 8 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 4.5 * mm, "Verilumen Simulation Agent — User Guide")
        canvas.drawRightString(PAGE_W - MARGIN, 4.5 * mm, f"Page {doc_.page} of 2")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_decor, onLaterPages=add_page_decor)
    print(OUT)


if __name__ == "__main__":
    build()
