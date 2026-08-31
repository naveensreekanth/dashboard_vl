"""Generate PDF user guide for Post-process (ATE logs → next-chip recommend)."""

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "images"
OUT = ROOT / "Post_Process_User_Guide.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 12 * mm

INK = HexColor("#1a1f26")
MUTED = HexColor("#5a6570")
ACCENT = HexColor("#1f6b4a")


def style_sheet():
    base = getSampleStyleSheet()
    return {
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
            spaceBefore=2.5 * mm,
            spaceAfter=1.2 * mm,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.4,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=1.4 * mm,
        ),
        "step": ParagraphStyle(
            "step",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=INK,
            leftIndent=2 * mm,
            spaceAfter=0.7 * mm,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=INK,
            leftIndent=3 * mm,
            spaceAfter=0.5 * mm,
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


def fit_image(path: Path, max_w, max_h):
    if not path.exists():
        return None
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def add_img(story, name, max_w, max_h, gap=1.2):
    img = fit_image(IMG / name, max_w, max_h)
    if img:
        story.append(img)
        story.append(Spacer(1, gap * mm))


def build():
    styles = style_sheet()
    content_w = PAGE_W - 2 * MARGIN

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(
            PAGE_W / 2,
            7 * mm,
            f"Post-process User Guide  ·  page {doc.page}",
        )
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=10 * mm,
        bottomMargin=14 * mm,
        title="Post-process User Guide — Verilumen",
        author="Verilumen Simulation Agent",
    )

    story = []

    story.append(Paragraph("Post-process User Guide", styles["title"]))
    story.append(
        Paragraph(
            "ATE log analysis → next-chip pattern recommendation  ·  Verilumen Simulation Agent",
            styles["subtitle"],
        )
    )

    story.append(
        Paragraph(
            "This guide explains the <b>Post-process</b> window. Use it <b>after</b> ATE testing: "
            "take the fail log from <b>chip N</b> (one log = one die/chip), analyze which patterns "
            "failed, and get an <b>ordered list</b> of patterns to run on <b>chip N+1</b>.",
            styles["body"],
        )
    )

    story.append(Paragraph("How the stages fit together", styles["h"]))
    story.append(
        Paragraph(
            "<b>Pre-process</b> (before ATE): STIL → keep/skip prediction. "
            "<b>ATE</b>: run patterns, produce a fail/pass log. "
            "<b>Post-process</b> (after ATE): log + last pre-process lists → ordered patterns for the next chip. "
            "Rule of thumb: <b>1 ATE log file = 1 die = 1 chip</b>.",
            styles["body"],
        )
    )

    story.append(Paragraph("Recommended steps", styles["h"]))
    for i, t in enumerate(
        [
            "Finish <b>Pre-process</b> (upload STIL → <b>Run live simulation</b> → wait until <b>Done</b>).",
            "Open the <b>Post-process</b> tab.",
            "Add the ATE log from the chip you just tested (<b>Add log file</b> or <b>Add log folder</b>).",
            "Click <b>Analyze &amp; recommend next chip</b>.",
            "Review the summary cards and the <b>Recommended patterns for next chip</b> table.",
            "Optionally click <b>Copy ordered list</b> and use that order on the next chip.",
        ],
        start=1,
    ):
        story.append(Paragraph(f"<b>Step {i}.</b> {t}", styles["step"]))

    story.append(
        Paragraph(
            "Analysis runs <b>locally in your browser</b>. Log files stay on your PC.",
            styles["body"],
        )
    )

    # ---- Full page ----
    story.append(Paragraph("1. Full Post-process page", styles["h"]))
    add_img(story, "post-00-full.png", content_w, 78 * mm)
    story.append(
        Paragraph(
            "The page has a <b>header</b> (title + Pre/Post switch), a <b>left panel</b> "
            "(add logs and run analysis), and a <b>main area</b> (status, summary metrics, "
            "and the next-chip recommend table).",
            styles["body"],
        )
    )

    story.append(PageBreak())

    # ---- Switch ----
    story.append(Paragraph("2. Switching to Post-process", styles["h"]))
    add_img(story, "post-01-switch.png", content_w, 42 * mm)
    story.append(
        Paragraph(
            "Under the title: <b>Pre-process</b> = STIL → pattern prediction (before ATE); "
            "<b>Post-process</b> = ATE logs → fail analysis (after ATE). "
            "You need a finished pre-process run so the tool knows the full pattern set "
            "(keep + skip, usually 1000 patterns).",
            styles["body"],
        )
    )

    # ---- Sidebar ----
    story.append(Paragraph("3. Left panel — Fail logs", styles["h"]))
    add_img(story, "post-02-sidebar.png", content_w * 0.95, 72 * mm)
    story.append(
        Paragraph(
            "Use <b>Add log folder</b> or <b>Add log files</b>. For the usual flow "
            "(1 chip → 1 log), one log file is enough. "
            "<b>Analyze &amp; recommend next chip</b> parses pass/fail and channel "
            "<font face='Courier'>STATUS:F</font>, matches patterns to the last pre-process "
            "keep/skip lists, and builds the ordered next-chip list. "
            "<b>Clear logs</b> removes selected files and clears the current analysis. "
            "Until you analyze, the main pane shows a short checklist.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Status may show keep/skip counts from pre-process (e.g. keep 600 · skip 400). "
            "That is context only — analysis still uses <b>all</b> patterns, not only the keep list.",
            styles["body"],
        )
    )

    story.append(PageBreak())

    # ---- Summary ----
    story.append(Paragraph("4. Status line and summary cards", styles["h"]))
    add_img(story, "post-03-summary.png", content_w, 48 * mm)
    story.append(
        Paragraph(
            "The status line reports how many patterns to run next, how many failed in the log, "
            "and how many pass-only patterns were skipped. "
            "Summary cards: <b>Log folders (dies)</b> (1 file → usually 1); "
            "<b>high recommended patterns from pre-process</b> (fails among kept patterns); "
            "<b>low risk patterns from pre-process</b> (fails among skipped patterns); "
            "<b>Files parsed</b>. "
            "These two fail cards compare silicon results to the pre-process split; they do not "
            "limit the next-chip list to keep-only. A skip pattern that failed can still be recommended "
            "(shown later with a <b>was skip</b> badge).",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Log labels are often 1-based (P1…P1000); the dashboard uses ids 0…999 "
            "(log Pn → id n−1). Fail = any channel <font face='Courier'>STATUS:F</font>.",
            styles["body"],
        )
    )

    # ---- Recommend metrics ----
    story.append(Paragraph("5. Recommended patterns — header and metrics", styles["h"]))
    add_img(story, "post-04-recommend-metrics.png", content_w, 52 * mm)
    story.append(
        Paragraph(
            "This is the main deliverable. Help text: <b>1 log = 1 die/chip</b>; every pattern "
            "that failed is included; order by severity (channels → fail rate → fails); "
            "pass-only patterns are skipped. "
            "<b>Copy ordered list</b> copies pattern ids in run order. "
            "Metrics: <b>Recommend</b> (to run / total), <b>Failed in logs</b>, "
            "<b>Log pass → skip</b>, and <b>Failed patterns</b>.",
            styles["body"],
        )
    )

    story.append(PageBreak())

    # ---- Table ----
    story.append(Paragraph("6. Recommend table (ordered list)", styles["h"]))
    add_img(story, "post-05-recommend-table.png", content_w, 70 * mm)
    story.append(
        Paragraph(
            "Each row is one pattern for the next chip, in suggested run order. "
            "Columns: <b>Order</b>, <b>Pattern</b> (badge <b>was skip</b> = pre-process skip list "
            "but failed on silicon), <b>Tier</b> (<b>log fail</b> = failed in the ATE log), "
            "<b>Fail rate</b> = fails÷(fails+passes), <b>Fails</b>, and <b>Why</b>.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "With one die, order among failers is: more failing channels → higher fail rate → "
            "more fails → tie-breaks. If many rows look identical (all 100% fail rate, 1 fail), "
            "they are effectively a tie. "
            "Fail rate 100% on one die means fails/(fails+passes) = 1/(1+0) — failed on this chip, "
            "not “fails forever on every chip.”",
            styles["body"],
        )
    )

    story.append(Paragraph("7. Logic summary", styles["h"]))
    for line in [
        "Chip N ATE log → parse each pattern (pass / fail / channel STATUS:F).",
        "Failed ≥ 1 time → INCLUDE (order by severity).",
        "Passed only → SKIP for next chip.",
        "Ordered list → run on Chip N+1.",
    ]:
        story.append(Paragraph(f"• {line}", styles["bullet"]))

    story.append(Paragraph("8. Quick checklist", styles["h"]))
    for i, t in enumerate(
        [
            "Pre-process: STIL → Run live simulation → Done",
            "Open <b>Post-process</b>",
            "<b>Add log files</b> (1 log = 1 die)",
            "<b>Analyze &amp; recommend next chip</b>",
            "Read summary cards (high recommended / low risk fails from pre-process)",
            "Use the recommend table order; <b>Copy ordered list</b> for the next chip",
        ],
        start=1,
    ):
        story.append(Paragraph(f"<b>{i}.</b> {t}", styles["step"]))

    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "Source: docs/POST_PROCESS_GUIDE.md  ·  Screenshots: docs/images/post-*.png",
            styles["footer"],
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
