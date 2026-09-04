from __future__ import annotations

from datetime import date
from pathlib import Path
import shutil

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    Image as RLImage,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output" / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "NPEC_Labeling_Tool_Application_Guide.pdf"
ALIAS_OUTPUT_PDF = OUTPUT_DIR / "NPEC_Labeling_Tool_Comprehensive_Manual.pdf"
ASSETS_DIR = ROOT_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "NPEC-logo-black.png"
MARK_PATH = ASSETS_DIR / "npec-app-icon.png"


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "GuideTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            textColor=colors.HexColor("#143652"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "GuideSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#3A4E66"),
            spaceAfter=8,
        ),
        "meta": ParagraphStyle(
            "GuideMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#5B6E84"),
            spaceAfter=2,
        ),
        "h1": ParagraphStyle(
            "GuideH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#143652"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "GuideH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1F4D72"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "GuideBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#1F2A35"),
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "GuideBullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            leftIndent=14,
            textColor=colors.HexColor("#1F2A35"),
            spaceAfter=1,
        ),
        "note": ParagraphStyle(
            "GuideNote",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#4C6078"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "GuideSmall",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#607487"),
            spaceAfter=3,
        ),
        "chip": ParagraphStyle(
            "GuideChip",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.4,
            leading=10.5,
            textColor=colors.HexColor("#1D4260"),
            alignment=1,
            spaceAfter=0,
        ),
        "card_title": ParagraphStyle(
            "GuideCardTitle",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.8,
            leading=10.8,
            textColor=colors.HexColor("#36516A"),
            spaceAfter=1,
        ),
        "card_value": ParagraphStyle(
            "GuideCardValue",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=16,
            textColor=colors.HexColor("#143652"),
            spaceAfter=1,
        ),
    }


def _bullets(items: list[str], style: ParagraphStyle, numbered: bool = False) -> ListFlowable:
    bullet_type = "1" if numbered else "bullet"
    return ListFlowable(
        [ListItem(Paragraph(text, style)) for text in items],
        bulletType=bullet_type,
        leftIndent=18,
        bulletFontName="Helvetica",
        bulletFontSize=9,
        bulletOffsetY=2,
        spaceBefore=2,
        spaceAfter=6,
    )


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setStrokeColor(colors.HexColor("#C8D4DF"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, page_height - 1.7 * cm, page_width - doc.rightMargin, page_height - 1.7 * cm)
    canvas.line(doc.leftMargin, 1.45 * cm, page_width - doc.rightMargin, 1.45 * cm)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5B6E84"))
    canvas.drawString(doc.leftMargin, page_height - 1.35 * cm, "NPEC Labeling Tool - Comprehensive User Manual")
    canvas.drawRightString(page_width - doc.rightMargin, page_height - 1.35 * cm, f"Page {canvas.getPageNumber()}")

    canvas.drawString(doc.leftMargin, 1.05 * cm, "Generated by NPEC Labeling Tool documentation pipeline")
    canvas.drawRightString(page_width - doc.rightMargin, 1.05 * cm, date.today().isoformat())
    canvas.restoreState()


def _tab_table(styles: dict[str, ParagraphStyle]) -> Table:
    cell = ParagraphStyle(
        "TabMapCell",
        parent=styles["body"],
        fontName="Helvetica",
        fontSize=8.6,
        leading=10.8,
        spaceAfter=0,
    )
    head = ParagraphStyle(
        "TabMapHead",
        parent=styles["body"],
        fontName="Helvetica-Bold",
        fontSize=8.8,
        leading=10.8,
        textColor=colors.white,
        alignment=1,
        spaceAfter=0,
    )

    rows = [
        [Paragraph("Tab", head), Paragraph("Use This When", head), Paragraph("Primary Outputs", head)],
        [Paragraph("1) Labeling", cell), Paragraph("Creating or editing pixel-accurate ground truth.", cell), Paragraph("Indexed masks, class layers, project state.", cell)],
        [Paragraph("2) Segmentation Pipeline", cell), Paragraph("Comparing model prediction vs ground truth and reviewing quality.", cell), Paragraph("Predictions, disagreement maps, plant tracking metrics, root decomposition overlay.", cell)],
        [Paragraph("3) Random Forest Segmentation", cell), Paragraph("Fast minimal-seed segmentation before manual cleanup.", cell), Paragraph("RF predictions, refined masks, training-ready labels.", cell)],
        [Paragraph("4) Patching", cell), Paragraph("Fixing disagreement regions rapidly.", cell), Paragraph("Corrected labels with reduced manual scope.", cell)],
        [Paragraph("5) Training", cell), Paragraph("Training or fine-tuning U-Net style models.", cell), Paragraph(".keras/.h5 model, optional .tflite export, training logs.", cell)],
        [Paragraph("6) Timelapse", cell), Paragraph("Visual QA over time series frames.", cell), Paragraph("1080p MP4 overlay timelapse.", cell)],
        [Paragraph("7) Analytics", cell), Paragraph("Temporal growth quantification and benchmarking.", cell), Paragraph("CSV/JSON/RSML/QC MP4, T50/TMGR, per-track growth curves.", cell)],
        [Paragraph("8) Additional Sensors", cell), Paragraph("Aligning fluorescence or hyperspectral with base grayscale images.", cell), Paragraph("Corrected modalities, calibrated overlays, saved QC images.", cell)],
        [Paragraph("9) Foundation Segmentation", cell), Paragraph("Interactive click/box/scribble segmentation with foundation backends.", cell), Paragraph("Prompt masks, assisted predictions, interaction logs.", cell)],
        [Paragraph("10) Consolidated Measurements", cell), Paragraph("Merging per-run measurement files and creating plot timelapses.", cell), Paragraph("Consolidated tables and metric MP4 plots.", cell)],
    ]

    table = Table(rows, colWidths=[3.2 * cm, 6.5 * cm, 7.9 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4260")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F4F8FB"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#B4C6D5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _troubleshooting_table(styles: dict[str, ParagraphStyle]) -> Table:
    cell = ParagraphStyle(
        "TroubleCell",
        parent=styles["body"],
        fontName="Helvetica",
        fontSize=8.4,
        leading=10.6,
        spaceAfter=0,
    )
    head = ParagraphStyle(
        "TroubleHead",
        parent=styles["body"],
        fontName="Helvetica-Bold",
        fontSize=8.7,
        leading=10.8,
        textColor=colors.white,
        alignment=1,
        spaceAfter=0,
    )

    rows = [
        [Paragraph("Symptom", head), Paragraph("Likely Cause", head), Paragraph("Action", head)],
        [Paragraph("No module named patchify / rich.progress", cell), Paragraph("Runtime package collection missing in frozen build.", cell), Paragraph("Rebuild with provided scripts. Ensure --collect-all patchify and --collect-all rich are present.", cell)],
        [Paragraph("imageio metadata not found or MP4 export fails", cell), Paragraph("imageio/imageio-ffmpeg metadata not bundled.", cell), Paragraph("Use current build scripts that include --copy-metadata imageio and imageio_ffmpeg.", cell)],
        [Paragraph("TensorFlow required to load .h5/.keras", cell), Paragraph("Packaged runtime does not include TF or mismatch environment.", cell), Paragraph("Select External TF Python path that points to your NPEC_GPU environment python executable.", cell)],
        [Paragraph("Conv2DTranspose groups argument error", cell), Paragraph("Model serialized with incompatible Keras/TensorFlow version semantics.", cell), Paragraph("Use packaged compatibility loader path in Pipeline tab or retrain/export model in compatible TF stack.", cell)],
        [Paragraph("Windows build enters Python prompt", cell), Paragraph("python command called without -m venv due script mismatch or shell alias issue.", cell), Paragraph("Run build_windows.cmd from a clean Command Prompt and ensure latest build_windows.ps1 is present.", cell)],
        [Paragraph("Painting lag or UI stalls", cell), Paragraph("Very large image with frequent full view refresh.", cell), Paragraph("Use ROI workflow, keep preview contrast moderate, and run batch jobs from lazy folder mode.", cell)],
    ]

    table = Table(rows, colWidths=[4.3 * cm, 5.7 * cm, 7.6 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4260")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F4F8FB"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#B4C6D5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _deployment_matrix_table(styles: dict[str, ParagraphStyle]) -> Table:
    cell = ParagraphStyle(
        "DeployCell",
        parent=styles["body"],
        fontName="Helvetica",
        fontSize=8.6,
        leading=10.7,
        spaceAfter=0,
    )
    head = ParagraphStyle(
        "DeployHead",
        parent=styles["body"],
        fontName="Helvetica-Bold",
        fontSize=8.8,
        leading=10.9,
        textColor=colors.white,
        alignment=1,
        spaceAfter=0,
    )

    rows = [
        [Paragraph("Target", head), Paragraph("Package", head), Paragraph("Install", head), Paragraph("Notes", head)],
        [Paragraph("macOS (Apple Silicon)", cell), Paragraph(".app + .dmg", cell), Paragraph("Open .dmg, drag app to Applications.", cell), Paragraph("If blocked on first open: right-click -> Open.", cell)],
        [Paragraph("Windows (RTX / CPU)", cell), Paragraph("Windows source zip -> built exe", cell), Paragraph("Run build_windows.cmd in extracted folder.", cell), Paragraph("Use External TF Python path when TensorFlow is managed separately.", cell)],
        [Paragraph("Lab deployment", cell), Paragraph("Versioned zip archives", cell), Paragraph("Distribute with checksum manifest file.", cell), Paragraph("Keep per-release folders immutable for reproducibility.", cell)],
    ]

    table = Table(rows, colWidths=[3.4 * cm, 4.2 * cm, 4.9 * cm, 4.9 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4260")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F4F8FB"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#B4C6D5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _brand_header_panel(styles: dict[str, ParagraphStyle]) -> Table:
    left_block = Paragraph(
        "<b>NPEC Labeling Tool</b><br/>"
        "Production guide for labeling, model validation, analytics, and deployment.<br/>"
        "<font size='8' color='#5B6E84'>Designed for multi-user macOS + Windows distribution.</font>",
        styles["body"],
    )

    if LOGO_PATH.exists():
        logo = RLImage(str(LOGO_PATH))
        logo.drawHeight = 1.4 * cm
        logo.drawWidth = 4.8 * cm
    else:
        logo = Paragraph("NPEC", styles["h1"])

    panel = Table(
        [[left_block, logo]],
        colWidths=[12.7 * cm, 4.7 * cm],
    )
    panel.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF2F8")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#B7CBDB")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ]
        )
    )
    return panel


def _kpi_cards(styles: dict[str, ParagraphStyle]) -> Table:
    cards = [
        [
            Paragraph("Workflow Tabs", styles["card_title"]),
            Paragraph("10", styles["card_value"]),
            Paragraph("Labeling to consolidation", styles["small"]),
        ],
        [
            Paragraph("Export Outputs", styles["card_title"]),
            Paragraph("8+", styles["card_value"]),
            Paragraph("PNG, CSV, JSON, RSML, MP4, .keras, .h5, .tflite", styles["small"]),
        ],
        [
            Paragraph("Platforms", styles["card_title"]),
            Paragraph("2", styles["card_value"]),
            Paragraph("macOS and Windows final builds", styles["small"]),
        ],
        [
            Paragraph("Core Loop", styles["card_title"]),
            Paragraph("5-step", styles["card_value"]),
            Paragraph("Label, infer, patch, train, benchmark", styles["small"]),
        ],
    ]

    table = Table(cards, colWidths=[4.3 * cm, 4.3 * cm, 4.3 * cm, 4.3 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F8FB")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#C2D2DF")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D3DFE8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _workflow_diagram() -> Drawing:
    drawing = Drawing(17.2 * cm, 3.8 * cm)
    box_w = 3.15 * cm
    box_h = 1.15 * cm
    y = 1.8 * cm
    step_x = [0.1 * cm, 3.55 * cm, 7.0 * cm, 10.45 * cm, 13.9 * cm]
    labels = [
        "1) Label",
        "2) Segment",
        "3) Patch",
        "4) Train",
        "5) Benchmark",
    ]
    fills = ["#E9F3FB", "#EAF8F1", "#FFF6E8", "#F0EEFA", "#EAF2F8"]
    strokes = ["#2A628E", "#2F7A57", "#AF6D1A", "#5D4AA6", "#2A628E"]

    for idx, x in enumerate(step_x):
        drawing.add(Rect(x, y, box_w, box_h, fillColor=colors.HexColor(fills[idx]), strokeColor=colors.HexColor(strokes[idx]), strokeWidth=1.0, rx=4, ry=4))
        drawing.add(String(x + 0.2 * cm, y + 0.66 * cm, labels[idx], fontName="Helvetica-Bold", fontSize=8.3, fillColor=colors.HexColor("#143652")))
        if idx < len(step_x) - 1:
            x1 = x + box_w
            x2 = step_x[idx + 1] - 0.05 * cm
            drawing.add(Line(x1, y + 0.58 * cm, x2, y + 0.58 * cm, strokeColor=colors.HexColor("#56748E"), strokeWidth=1.1))
            drawing.add(Line(x2 - 0.16 * cm, y + 0.68 * cm, x2, y + 0.58 * cm, strokeColor=colors.HexColor("#56748E"), strokeWidth=1.1))
            drawing.add(Line(x2 - 0.16 * cm, y + 0.48 * cm, x2, y + 0.58 * cm, strokeColor=colors.HexColor("#56748E"), strokeWidth=1.1))

    drawing.add(String(0.1 * cm, 0.62 * cm, "Closed loop improvement protocol: iterate until disagreement and temporal instability drop.", fontName="Helvetica", fontSize=8.2, fillColor=colors.HexColor("#4B5F73")))
    return drawing


def _adoption_chart() -> Drawing:
    drawing = Drawing(17.2 * cm, 5.1 * cm)
    chart = VerticalBarChart()
    chart.x = 1.0 * cm
    chart.y = 0.9 * cm
    chart.height = 3.4 * cm
    chart.width = 14.8 * cm
    chart.data = [[32, 48, 61, 74, 82]]
    chart.categoryAxis.categoryNames = ["Raw labels", "After pipeline", "After patching", "After training", "Production"]
    chart.categoryAxis.labels.boxAnchor = "n"
    chart.categoryAxis.labels.angle = 0
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7.5
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 100
    chart.valueAxis.valueStep = 20
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7.2
    chart.bars[0].fillColor = colors.HexColor("#2A628E")
    chart.bars[0].strokeColor = colors.HexColor("#1F4D72")
    chart.barWidth = 0.9 * cm
    chart.groupSpacing = 0.5 * cm
    drawing.add(chart)
    drawing.add(String(1.0 * cm, 4.55 * cm, "Example quality progression (visual guide)", fontName="Helvetica-Bold", fontSize=8.8, fillColor=colors.HexColor("#143652")))
    drawing.add(String(1.0 * cm, 0.22 * cm, "Use as communication graphic for expected model quality gains across the loop.", fontName="Helvetica", fontSize=7.7, fillColor=colors.HexColor("#4B5F73")))
    return drawing


def _append_tab_section(
    story: list,
    styles: dict[str, ParagraphStyle],
    title: str,
    goal: str,
    controls: list[str],
    workflow: list[str],
    outputs: list[str],
) -> None:
    story.append(Paragraph(title, styles["h1"]))
    story.append(Paragraph(f"<b>Goal:</b> {goal}", styles["body"]))
    story.append(Paragraph("Key controls", styles["h2"]))
    story.append(_bullets(controls, styles["bullet"]))
    story.append(Paragraph("Recommended workflow", styles["h2"]))
    story.append(_bullets(workflow, styles["bullet"], numbered=True))
    story.append(
        KeepTogether(
            [
                Paragraph("Typical outputs", styles["h2"]),
                _bullets(outputs, styles["bullet"]),
            ]
        )
    )


def build_pdf(output_pdf: Path = OUTPUT_PDF) -> Path:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    styles = _build_styles()
    today = date.today().isoformat()

    story: list = []

    story.extend(
        [
            _brand_header_panel(styles),
            Spacer(1, 0.25 * cm),
            Paragraph("NPEC Labeling Tool", styles["title"]),
            Paragraph(
                "Comprehensive user manual for annotation, model testing, patching, training, analytics, and deployment workflows.",
                styles["subtitle"],
            ),
            Paragraph(
                "Designed and created by Vinicius Lube. Extensive application testing and feedback by Jason van Hamond. February 2026, Utrecht University.",
                styles["meta"],
            ),
            Paragraph(f"Manual build date: {today}", styles["meta"]),
            Spacer(1, 0.35 * cm),
            Paragraph("At a glance", styles["h1"]),
            _kpi_cards(styles),
            Spacer(1, 0.2 * cm),
            _workflow_diagram(),
            Spacer(1, 0.15 * cm),
            _adoption_chart(),
            Spacer(1, 0.2 * cm),
            Paragraph("What this manual covers", styles["h1"]),
            _bullets(
                [
                    "How to operate every tab in the desktop app with practical settings and expected outputs.",
                    "How to build a closed loop segmentation improvement process from first labels to retraining.",
                    "How to export results for model development, quality control, publication, and archiving.",
                    "How to troubleshoot packaging, runtime dependencies, and model compatibility issues on macOS and Windows.",
                ],
                styles["bullet"],
            ),
            Paragraph("Quick start in 15 minutes", styles["h1"]),
            _bullets(
                [
                    "Load images in 1) Labeling, create classes, and draw initial masks on representative frames.",
                    "Open 2) Segmentation Pipeline, load your model, run current frame inference, and inspect disagreement.",
                    "If using PyPhenotyper backend, enable tracking and run dataset inference for Plant Tracks metrics.",
                    "Open 4) Patching to correct disagreement regions and enforce temporal consistency.",
                    "Run 5) Training with augmentation to produce an updated model.",
                    "Validate with 6) Timelapse and 7) Analytics, then export masks and metrics.",
                ],
                styles["bullet"],
                numbered=True,
            ),
            Paragraph("Tab map", styles["h1"]),
            _tab_table(styles),
            Spacer(1, 0.2 * cm),
            Paragraph(
                "Tip: the Segmentation Pipeline now includes a root decomposition overlay so lateral roots are visible separately from the main root.",
                styles["note"],
            ),
            Paragraph("Interface overview", styles["h1"]),
            Paragraph("Main layout", styles["h2"]),
            _bullets(
                [
                    "Left sidebar: dataset timeline, project save/load, time-series loader, and navigation.",
                    "Top toolbar: tool mode, zoom controls, brush size, overlay intensity, tablet pressure toggle, and shortcut map.",
                    "Center workspace: tab content with large canvases and split compare regions.",
                    "Right dock: class list with color controls, drag reorder, and layer import options.",
                    "Status bar: current frame, tool/class status, and zoom percentage.",
                ],
                styles["bullet"],
            ),
            Paragraph("Global shortcuts", styles["h2"]),
            _bullets(
                [
                    "B, E, R for Brush, Eraser, Razor.",
                    "[, ], -, =, and Shift+[ or Shift+] for brush size adjustment.",
                    "Alt + mouse wheel for fast brush size changes while drawing.",
                    "Mouse wheel for zoom, Space + drag for pan, and Fit button to reset framing.",
                    "A and D to move previous/next frame in timeline.",
                    "Undo and redo: Cmd+Z and Cmd+Shift+Z on macOS, Ctrl+Z and Ctrl+Shift+Z or Ctrl+Y on Windows/Linux.",
                    "F1 opens the shortcut map from anywhere in the app.",
                ],
                styles["bullet"],
            ),
            Paragraph("Project persistence", styles["h2"]),
            _bullets(
                [
                    "Use Save Project and Load Project to persist dataset state, masks, predictions, series grouping, and model paths.",
                    "The app prompts for save confirmation on close if there are unsaved edits.",
                    "Projects are portable when image paths are still valid on the target machine.",
                ],
                styles["bullet"],
            ),
        ]
    )

    _append_tab_section(
        story,
        styles,
        "1) Labeling",
        "Create high quality pixel-wise ground truth with low latency drawing tools.",
        [
            "Load folder, load files, and Open Time Series to organize large experiments.",
            "Class management: add, remove, rename, recolor, and drag reorder class priority.",
            "Layer import supports standard mask files including .jxl mask image loading paths.",
            "Brush/eraser/razor with visible perimeter cursor, tablet pressure scaling, and fast zoom/pan.",
            "Undo and redo stroke history for drawing and erasing actions.",
        ],
        [
            "Create or verify classes and assign each class a distinct color.",
            "Start broad labeling at low zoom, then refine boundaries at high zoom.",
            "Use eraser and razor for boundary cleanup and narrow structures.",
            "Save project regularly, especially before model testing runs.",
        ],
        [
            "Indexed masks for direct model training pipelines.",
            "Class-wise layered masks for audit and correction workflows.",
            "Reusable project state for continuation sessions.",
        ],
    )

    _append_tab_section(
        story,
        styles,
        "2) Segmentation Pipeline",
        "Benchmark segmentation quality and inspect model behavior frame by frame and across the dataset.",
        [
            "Backend selector: Standard (.tflite/.h5/.keras) or PyPhenotyper dual-model backend.",
            "Run Current and Run Dataset inference actions with progress feedback.",
            "Lazy folder mode: select time-series input/output directories for disk-based segmentation without loading all frames into memory.",
            "Three synchronized previews: Ground Truth, Model Prediction, Disagreement.",
            "Split Compare slider for GT vs prediction side-by-side in one viewport.",
            "Plant Tracks sortable table for per-plant growth summary across frames.",
            "Main root and lateral root decomposition controls: Show Main Root, Show Lateral Roots, Overlay alpha.",
        ],
        [
            "Choose backend and set model paths before running inference.",
            "Run current frame to sanity check classes, scaling, and output shape.",
            "Run dataset to populate per-frame predictions and tracking metadata.",
            "Review disagreement percentage and per-class IoU/Dice.",
            "Enable lateral root visualization to confirm branch visibility and continuity.",
            "Export plant metrics CSV after PyPhenotyper dataset runs.",
        ],
        [
            "Prediction masks attached to timeline frames.",
            "Disagreement overlays for rapid QA targeting.",
            "Plant metrics CSV and tracking metadata.",
            "Main/lateral decomposition metrics in the metrics panel.",
        ],
    )

    story.append(PageBreak())

    _append_tab_section(
        story,
        styles,
        "3) Random Forest Segmentation",
        "Perform click-efficient minimal seeding followed by manual refinement before deep model training.",
        [
            "RF parameters: sigma, trees, depth, refinement iterations.",
            "Seed tools with class-specific drawing and optional ROI crop to limit compute area.",
            "Batch mode with optional seed propagation across similarly shaped frames.",
            "Refinement canvas to clean RF output and convert to training-grade masks.",
            "Direct handoff actions for export and U-Net training launch.",
        ],
        [
            "Seed only essential regions of root/shoot classes.",
            "Tune sigma or run Optimize Sigma for difficult contrast scenes.",
            "Run RF on current frame or batch, then inspect output quality.",
            "Adopt output into refinement stage and clean artifacts.",
            "Export refined masks or continue into Training tab.",
        ],
        [
            "Fast initial segmentation masks for large frame sets.",
            "Curated refinement masks suitable for supervised learning.",
            "Batch run logs and optional per-image status records.",
        ],
    )

    _append_tab_section(
        story,
        styles,
        "4) Patching",
        "Accelerate correction by focusing only on disagreement regions.",
        [
            "Restrict edits to disagreement toggle.",
            "Adopt Prediction action to auto-fill disagreement with model output.",
            "Clear Disagreement action to erase weak regions for manual relabeling.",
            "Full brush/eraser/razor support with same performance profile as labeling.",
        ],
        [
            "Run model inference first in Segmentation Pipeline.",
            "Open Patching and keep restriction enabled for efficient QA.",
            "Correct biologically implausible areas first, then boundary detail.",
            "Re-run inference and repeat until disagreement falls to acceptable range.",
        ],
        [
            "Quality-improved labels with targeted manual effort.",
            "Reduced correction time compared with full-frame relabeling.",
        ],
    )

    _append_tab_section(
        story,
        styles,
        "5) Training",
        "Train from scratch or fine tune with robust augmentation and GPU-aware execution.",
        [
            "Core hyperparameters: patch size, stride, batch size, epochs, learning rate, validation split.",
            "Architecture controls: base filters, dropout, freeze fraction for fine-tuning.",
            "Device mode: auto, GPU only, CPU only (supports RTX and Apple Silicon workflows).",
            "Augmentation controls: flips, rotate90, brightness, contrast, noise, gamma, blur, and replication.",
            "Optional TFLite export after training.",
        ],
        [
            "Start with conservative patch settings and verify memory footprint.",
            "Enable augmentation profile that matches observed domain shifts.",
            "Run short sanity training first, then full epochs for final model.",
            "Benchmark new model in Segmentation Pipeline and Analytics.",
        ],
        [
            "Trained .keras/.h5 model and optional .tflite model.",
            "Training logs and convergence metrics in the UI.",
        ],
    )

    _append_tab_section(
        story,
        styles,
        "6) Timelapse",
        "Validate temporal consistency and produce communication-ready QC videos.",
        [
            "Playback controls: frame slider, play/pause, and FPS.",
            "Overlay toggles for original image, labels, and predictions.",
            "Direct MP4 export at 1080p.",
        ],
        [
            "Run inference and optional patching first.",
            "Play through series to detect flicker, identity swaps, or growth discontinuities.",
            "Export MP4 for lab review and experiment reports.",
        ],
        [
            "1080p MP4 timelapse files for QA and presentations.",
        ],
    )

    _append_tab_section(
        story,
        styles,
        "7) Analytics",
        "Quantify growth trajectories, temporal stability, and segmentation benchmark quality.",
        [
            "Tracking configuration: auto vs manual ROI, component filters, smoothing controls.",
            "Scale and timing controls: pixel size (mm/px), timestep hours.",
            "Benchmark metrics: macro Dice, IoU, Hausdorff, class-level statistics.",
            "QC rendering controls: overlay color/alpha, line thickness, text scale, contrast.",
            "Exports: CSV, JSON, RSML, and QC MP4.",
        ],
        [
            "Run dataset analytics after segmentation is available for target frames.",
            "Inspect per-track growth and branch/tip behavior for biological plausibility.",
            "Run benchmark against annotations to quantify model quality.",
            "Export structured outputs for downstream statistical analysis.",
        ],
        [
            "Track-level and frame-level growth tables.",
            "Summary metrics including T50 and TMGR when data supports it.",
            "Quality-control video and machine-readable analysis exports.",
        ],
    )

    _append_tab_section(
        story,
        styles,
        "8) Additional Sensors",
        "Fuse grayscale images with fluorescence/hyperspectral sources using optical correction profiles.",
        [
            "Load base grayscale image and sensor dataset (.hdr/.bil/.tar and image formats).",
            "Select or create optical correction JSON profile from Pipelines directories.",
            "Configure overlay alpha and inspect corrected channels side-by-side.",
            "Save corrected overlay for reporting or cross-modality QA.",
        ],
        [
            "Load base image first, then sensor data.",
            "Apply correction profile and inspect alignment quality.",
            "Tune profile values if drift or distortion remains.",
            "Save corrected overlay and include it in QA package.",
        ],
        [
            "Aligned fluorescence/hyperspectral overlays.",
            "Reusable optical correction JSON templates.",
        ],
    )

    _append_tab_section(
        story,
        styles,
        "9) Foundation Segmentation",
        "Use modern prompt-driven segmentation backends (clicks, boxes, scribbles) for fast annotation support.",
        [
            "Backend chooser with internal and external runner support.",
            "Prompt classes for positive/negative interactions.",
            "Box prompt mode, rosette seed priors, and next-click suggestion support.",
            "Batch assist mode and interaction/run log export.",
            "Apply predictions directly to labels for correction workflows.",
        ],
        [
            "Select output class and prompt mode.",
            "Add sparse clicks/scribbles or a box to localize target region.",
            "Run current prediction and inspect boundary quality.",
            "Use suggestion tool for hard regions, then apply to labels if acceptable.",
            "Export interaction logs for NoC and QA analytics.",
        ],
        [
            "Interactive segmentation masks integrated into existing class layers.",
            "Prompt interaction logs and run history CSV files.",
        ],
    )

    _append_tab_section(
        story,
        styles,
        "10) Consolidated Measurements",
        "Aggregate metrics from many segmentation runs and generate plot timelapse videos.",
        [
            "Scan root folder recursively for measurement files (CSV/XLSX/TSV and run details).",
            "Automatic series and petri dish metadata parsing from path and filename tokens.",
            "Preview consolidated table and select target metric.",
            "Generate metric timelapse MP4 from consolidated measurements.",
        ],
        [
            "Point to measurements root and output folder.",
            "Run consolidation and inspect warnings and detected columns.",
            "Export consolidated CSV/XLSX summary.",
            "Render metric timelapse plot for temporal interpretation.",
        ],
        [
            "Master consolidated measurement table and time-evolving metric plot videos (MP4).",
        ],
    )

    story.extend(
        [
            Paragraph("Closed loop protocol to improve segmentation models", styles["h1"]),
            _bullets(
                [
                    "Label representative frames and edge cases in 1) Labeling.",
                    "Evaluate in 2) Segmentation Pipeline, focusing on disagreement and class-wise metrics.",
                    "Correct only high-value regions in 4) Patching.",
                    "Train or fine tune in 5) Training with a documented augmentation profile.",
                    "Re-run 6) Timelapse and 7) Analytics to validate temporal stability.",
                    "Consolidate outputs in 10) Consolidated Measurements to monitor experiment-level trends.",
                ],
                styles["bullet"],
                numbered=True,
            ),
            Paragraph("Performance tuning and responsiveness", styles["h1"]),
            _bullets(
                [
                    "Use lazy folder segmentation for very large time-series datasets.",
                    "Use ROI-based workflows in RF and Pipeline tabs to reduce compute load.",
                    "Run long batch segmentation/training jobs while avoiding unnecessary full-tab refreshes.",
                    "Keep class count and overlay thickness reasonable during interactive painting.",
                    "Prefer dedicated GPU environments for .h5/.keras inference/training on Windows and high-end RTX systems.",
                ],
                styles["bullet"],
            ),
            Paragraph("Export and file formats", styles["h1"]),
            _bullets(
                [
                    "Masks: indexed PNG and class-layer masks for model training pipelines.",
                    "Projects: .oclp files storing timeline state, labels, predictions, and model/runtime paths.",
                    "Models: .tflite, .h5, .keras inputs and optional .tflite training output.",
                    "Metrics: CSV/JSON/RSML and MP4 outputs from analytics and consolidation tabs.",
                    "Videos: 1080p MP4 from Timelapse and QC exporters.",
                ],
                styles["bullet"],
            ),
            Paragraph("Troubleshooting quick reference", styles["h1"]),
            _troubleshooting_table(styles),
            Spacer(1, 0.2 * cm),
            Paragraph("Deployment notes (macOS and Windows)", styles["h1"]),
            _deployment_matrix_table(styles),
            Spacer(1, 0.15 * cm),
            _bullets(
                [
                    "macOS: distribute .dmg or zipped .app from output/mac artifacts.",
                    "Windows: build natively on Windows with build_windows.cmd (Python 3.11/3.12 recommended).",
                    "If TensorFlow is external, set External TF Python path in the app to your prepared environment executable.",
                    "For reproducibility, version model files with date, dataset subset, and augmentation profile.",
                ],
                styles["bullet"],
            ),
            Paragraph("Recommended data organization", styles["h1"]),
            _bullets(
                [
                    "Use one folder per experiment and one subfolder per time series or petri dish.",
                    "Keep original images immutable; write masks and run outputs to dedicated output folders.",
                    "Store project files near the dataset root but separate from raw images when possible.",
                    "Archive exported metrics and QC videos alongside model checkpoints for each training cycle.",
                ],
                styles["bullet"],
            ),
            Paragraph("Windows install quick reference", styles["h1"]),
            _bullets(
                [
                    "Unzip the windows source package on a Windows machine.",
                    "Open Command Prompt in the extracted resources folder.",
                    "Run build_windows.cmd (or build_windows.ps1 -PythonVersion 3.12).",
                    "Run dist-win/NPEC Labeling Tool/NPEC Labeling Tool.exe after build completes.",
                    "If TensorFlow is external, configure External TF Python in the Segmentation Pipeline tab.",
                ],
                styles["bullet"],
                numbered=True,
            ),
            Paragraph("macOS first run notes", styles["h1"]),
            _bullets(
                [
                    "Open the distributed .dmg and drag the app to Applications.",
                    "If Gatekeeper warns on first launch, open via right click then Open.",
                    "For Apple Silicon training workloads, prefer dedicated environments that match your TensorFlow version.",
                    "Keep project files and output exports in writable user folders, not inside the app bundle.",
                ],
                styles["bullet"],
            ),
            Paragraph("Output artifact glossary", styles["h1"]),
            _bullets(
                [
                    "output/pdf: generated manuals and documentation exports.",
                    "output/mac: macOS .dmg and zipped app artifacts.",
                    "output/windows: Windows source/build bundles and packaging assets.",
                    "segmentation_masks folders: lazy pipeline masks and per-run detail tables.",
                    "analytics exports: CSV/JSON/RSML/QC MP4 from Analytics tab.",
                ],
                styles["bullet"],
            ),
        ]
    )

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2.15 * cm,
        bottomMargin=1.95 * cm,
        title="NPEC Labeling Tool Comprehensive Manual",
        author="NPEC Lab",
        subject="Full application manual for annotation, segmentation, training, and analytics workflows",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)

    if output_pdf != ALIAS_OUTPUT_PDF:
        try:
            shutil.copyfile(output_pdf, ALIAS_OUTPUT_PDF)
        except Exception:
            pass

    return output_pdf


if __name__ == "__main__":
    pdf_path = build_pdf()
    print(str(pdf_path))
