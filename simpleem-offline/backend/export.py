"""PDF and CSV export for meeting analysis reports.

- PDF: fpdf2 (pure-Python, no system dependencies)
- CSV: ZIP file with engagement.csv, transcript.csv, participants.csv,
       coaching.csv, correlations.csv, personality.csv, signal_snapshots.csv
- Chart: matplotlib engagement timeline embedded in PDF
"""

import csv
import io
import json
import logging
import os
import tempfile
import zipfile
from typing import Optional

import aiosqlite
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF

from .models import DB_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engagement chart generation
# ---------------------------------------------------------------------------

async def _generate_engagement_chart(video_id: str) -> str | None:
    """Generate engagement timeline chart as PNG, return file path or None."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT timestamp, engagement FROM emotions WHERE video_id = ? ORDER BY timestamp",
            (video_id,),
        )
        rows = await cursor.fetchall()
        if not rows or len(rows) < 2:
            return None

    timestamps = [r["timestamp"] / 60 for r in rows]  # convert to minutes
    engagement = [r["engagement"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.fill_between(timestamps, engagement, alpha=0.25, color='#8b5cf6')
    ax.plot(timestamps, engagement, color='#8b5cf6', linewidth=1.8)
    ax.set_xlabel('Time (min)', fontsize=8, color='#64748b')
    ax.set_ylabel('Engagement %', fontsize=8, color='#64748b')
    ax.set_ylim(0, 100)
    ax.set_facecolor('#f8fafc')
    fig.patch.set_facecolor('#f8fafc')
    ax.tick_params(colors='#64748b', labelsize=7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.spines['left'].set_color('#e2e8f0')
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return tmp.name


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

async def generate_csv_export(video_id: str) -> bytes:
    """Generate a ZIP file containing CSV exports of all meeting data.

    Returns bytes of the ZIP file.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Video info
        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        video = await cursor.fetchone()
        if not video:
            raise ValueError(f"Video {video_id} not found")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Participants CSV
            cursor = await db.execute(
                "SELECT * FROM participants WHERE video_id = ?", (video_id,)
            )
            rows = await cursor.fetchall()
            csv_str = _rows_to_csv(
                rows,
                ["name", "engagement_score", "sentiment_score", "speaking_pct",
                 "clarity_score", "rapport_score", "energy_score"],
            )
            zf.writestr("participants.csv", csv_str)

            # Emotions/Engagement CSV
            cursor = await db.execute(
                "SELECT * FROM emotions WHERE video_id = ? ORDER BY timestamp",
                (video_id,),
            )
            rows = await cursor.fetchall()
            csv_str = _rows_to_csv(
                rows,
                ["timestamp", "participant_id", "emotion", "confidence", "engagement"],
            )
            zf.writestr("engagement.csv", csv_str)

            # Transcript CSV
            cursor = await db.execute(
                "SELECT * FROM transcript_segments WHERE video_id = ? ORDER BY start_time",
                (video_id,),
            )
            rows = await cursor.fetchall()
            csv_str = _rows_to_csv(
                rows,
                ["start_time", "end_time", "speaker", "text"],
            )
            zf.writestr("transcript.csv", csv_str)

            # Flags CSV
            cursor = await db.execute(
                "SELECT * FROM flags WHERE video_id = ? ORDER BY timestamp",
                (video_id,),
            )
            rows = await cursor.fetchall()
            csv_str = _rows_to_csv(
                rows,
                ["timestamp", "type", "description", "severity"],
            )
            zf.writestr("flags.csv", csv_str)

            # Voice signals CSV
            cursor = await db.execute(
                "SELECT * FROM voice_signals WHERE video_id = ? ORDER BY timestamp",
                (video_id,),
            )
            rows = await cursor.fetchall()
            csv_str = _rows_to_csv(
                rows,
                ["timestamp", "tone", "pace", "energy", "dynamics"],
            )
            zf.writestr("voice_signals.csv", csv_str)

            # Word signals CSV
            cursor = await db.execute(
                "SELECT * FROM word_signals WHERE video_id = ? ORDER BY timestamp",
                (video_id,),
            )
            rows = await cursor.fetchall()
            csv_str = _rows_to_csv(
                rows,
                ["timestamp", "sentiment", "sentiment_score", "keywords", "key_phrases"],
            )
            zf.writestr("word_signals.csv", csv_str)

            # Coaching CSV
            cursor = await db.execute(
                """SELECT c.recommendation, c.category, c.priority, p.name as participant_name
                   FROM coaching c JOIN participants p ON c.participant_id = p.id
                   WHERE c.video_id = ? ORDER BY p.name, c.priority""",
                (video_id,),
            )
            rows = await cursor.fetchall()
            csv_str = _rows_to_csv(
                rows,
                ["participant_name", "recommendation", "category", "priority"],
            )
            zf.writestr("coaching.csv", csv_str)

            # Correlations CSV
            cursor = await db.execute(
                "SELECT * FROM correlations WHERE video_id = ?", (video_id,),
            )
            rows = await cursor.fetchall()
            csv_str = _rows_to_csv(
                rows,
                ["signal_pair", "pattern", "confidence", "implication"],
            )
            zf.writestr("correlations.csv", csv_str)

            # Personality CSV
            cursor = await db.execute(
                """SELECT ps.traits, ps.communication_style, ps.confidence_level,
                          p.name as participant_name
                   FROM personality_signals ps
                   JOIN participants p ON ps.participant_id = p.id
                   WHERE ps.video_id = ? ORDER BY p.name""",
                (video_id,),
            )
            rows = await cursor.fetchall()
            csv_str = _rows_to_csv(
                rows,
                ["participant_name", "traits", "communication_style", "confidence_level"],
            )
            zf.writestr("personality.csv", csv_str)

            # Signal snapshots CSV
            cursor = await db.execute(
                """SELECT timestamp, participant_id, emotion, engagement,
                          posture, openness, leaning, gestures, reactions
                   FROM signal_snapshots
                   WHERE video_id = ? ORDER BY timestamp""",
                (video_id,),
            )
            rows = await cursor.fetchall()
            csv_str = _rows_to_csv(
                rows,
                ["timestamp", "participant_id", "emotion", "engagement",
                 "posture", "openness", "leaning", "gestures", "reactions"],
            )
            zf.writestr("signal_snapshots.csv", csv_str)

            # Summary
            cursor = await db.execute(
                "SELECT * FROM summaries WHERE video_id = ?", (video_id,)
            )
            row = await cursor.fetchone()
            if row:
                summary_str = f"Summary\n{row['summary']}\n\nKey Topics\n{row['key_topics']}\n\nOverall Sentiment\n{row['overall_sentiment']}"
                zf.writestr("summary.txt", summary_str)

        return zip_buffer.getvalue()


# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

async def generate_pdf_report(video_id: str) -> bytes:
    """Generate a PDF report from meeting analysis data.

    Uses fpdf2 (pure-Python, no system dependencies).
    Returns bytes of the PDF file.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        cursor = await db.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
        video = await cursor.fetchone()
        if not video:
            raise ValueError(f"Video {video_id} not found")

        cursor = await db.execute(
            "SELECT * FROM participants WHERE video_id = ?", (video_id,)
        )
        participants = await cursor.fetchall()

        cursor = await db.execute(
            "SELECT * FROM summaries WHERE video_id = ?", (video_id,)
        )
        summary = await cursor.fetchone()

        cursor = await db.execute(
            "SELECT * FROM flags WHERE video_id = ? ORDER BY timestamp", (video_id,)
        )
        flags = await cursor.fetchall()

        cursor = await db.execute(
            "SELECT * FROM transcript_segments WHERE video_id = ? ORDER BY start_time",
            (video_id,),
        )
        transcript = await cursor.fetchall()

        # Coaching recommendations (joined with participant names)
        cursor = await db.execute(
            """SELECT c.*, p.name as participant_name
               FROM coaching c JOIN participants p ON c.participant_id = p.id
               WHERE c.video_id = ? ORDER BY p.name, c.priority""",
            (video_id,),
        )
        coaching = await cursor.fetchall()

        # Cross-signal correlations
        cursor = await db.execute(
            "SELECT * FROM correlations WHERE video_id = ?", (video_id,),
        )
        correlations = await cursor.fetchall()

    # Generate engagement chart (needs its own DB connection via helper)
    chart_path = await _generate_engagement_chart(video_id)

    return _build_pdf(
        video, participants, summary, flags, transcript,
        coaching, correlations, chart_path,
    )


def _build_pdf(
    video, participants, summary, flags, transcript,
    coaching, correlations, chart_path,
) -> bytes:
    """Build a styled PDF report using fpdf2."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Colors
    BLUE = (59, 130, 246)
    DARK = (30, 41, 59)
    GRAY = (100, 116, 139)
    LIGHT_BG = (241, 245, 249)

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 12, "InsightAI Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*BLUE)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Meta line
    duration_min = round(video["duration"] / 60, 1) if video["duration"] else 0
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GRAY)
    pdf.cell(
        0, 6,
        f'{video["name"]}  |  {duration_min} min  |  {len(participants)} participants  |  Score: {round(video["overall_score"])}%',
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(6)

    # Summary section
    _section_heading(pdf, "Summary", DARK)
    summary_text = summary["summary"] if summary else "No summary available"
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*DARK)
    pdf.set_fill_color(248, 250, 252)
    pdf.multi_cell(0, 5, summary_text, fill=True)
    pdf.ln(3)

    key_topics = json.loads(summary["key_topics"]) if summary else []
    if key_topics:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(28, 5, "Key Topics: ")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 5, ", ".join(key_topics), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Engagement Timeline Chart
    if chart_path:
        _section_heading(pdf, "Engagement Timeline", DARK)
        pdf.image(chart_path, x=10, w=190)
        pdf.ln(4)
        try:
            os.remove(chart_path)
        except OSError:
            pass

    # Participants table
    _section_heading(pdf, "Participants", DARK)
    headers = ["Name", "Engagement", "Clarity", "Rapport", "Energy", "Speaking"]
    col_w = [50, 28, 24, 24, 24, 24]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_text_color(71, 85, 105)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border="B", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK)
    for p in participants:
        pdf.cell(col_w[0], 6, str(p["name"]))
        pdf.cell(col_w[1], 6, f'{round(p["engagement_score"])}%')
        pdf.cell(col_w[2], 6, f'{round(p["clarity_score"])}%')
        pdf.cell(col_w[3], 6, f'{round(p["rapport_score"])}%')
        pdf.cell(col_w[4], 6, f'{round(p["energy_score"])}%')
        pdf.cell(col_w[5], 6, f'{round(p["speaking_pct"])}%')
        pdf.ln()
    pdf.ln(4)

    # Coaching Recommendations
    if coaching:
        _section_heading(pdf, "Coaching Recommendations", DARK)
        current_name = None
        for item in coaching:
            if item["participant_name"] != current_name:
                current_name = item["participant_name"]
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(*BLUE)
                pdf.cell(0, 7, current_name, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*DARK)
            rec = item["recommendation"]
            if len(rec) > 120:
                rec = rec[:117] + "..."
            pdf.cell(6, 5, chr(8226))  # bullet
            pdf.multi_cell(0, 5, rec)
        pdf.ln(4)

    # Cross-Signal Correlations
    if correlations:
        _section_heading(pdf, "Cross-Signal Correlations", DARK)
        for corr in correlations:
            signal_pair = json.loads(corr["signal_pair"]) if isinstance(corr["signal_pair"], str) else corr["signal_pair"]
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*BLUE)
            label = " \u2194 ".join(signal_pair) if isinstance(signal_pair, list) else str(signal_pair)
            pdf.cell(0, 6, label, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*DARK)
            pdf.multi_cell(0, 5, corr["pattern"])
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*GRAY)
            pdf.multi_cell(0, 4, f"Confidence: {round(corr['confidence'] * 100)}% \u2014 {corr['implication']}")
            pdf.ln(2)
        pdf.ln(4)

    # Critical Moments
    if flags:
        _section_heading(pdf, "Critical Moments", DARK)
        flag_cols = [22, 30, 138]
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(*LIGHT_BG)
        pdf.set_text_color(71, 85, 105)
        for i, h in enumerate(["Time", "Type", "Description"]):
            pdf.cell(flag_cols[i], 7, h, border="B", fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK)
        for f in flags:
            y_before = pdf.get_y()
            pdf.cell(flag_cols[0], 6, _format_time(f["timestamp"]))
            severity = f["severity"]
            if severity == "high":
                pdf.set_text_color(220, 38, 38)
            elif severity == "medium":
                pdf.set_text_color(217, 119, 6)
            else:
                pdf.set_text_color(22, 163, 74)
            pdf.cell(flag_cols[1], 6, f['type'])
            pdf.set_text_color(*DARK)
            desc = f["description"]
            if len(desc) > 90:
                desc = desc[:87] + "..."
            pdf.cell(flag_cols[2], 6, desc)
            pdf.ln()
        pdf.ln(4)

    # Transcript
    _section_heading(pdf, "Transcript", DARK)
    pdf.set_font("Helvetica", "", 9)
    for seg in transcript[:50]:
        ts = _format_time(seg["start_time"])
        speaker = seg["speaker"] or ""
        text = seg["text"] or ""
        pdf.set_text_color(*GRAY)
        pdf.cell(16, 5, ts)
        if speaker:
            pdf.set_text_color(*BLUE)
            pdf.cell(24, 5, speaker)
        pdf.set_text_color(*DARK)
        remaining_w = pdf.w - pdf.l_margin - pdf.r_margin - 16 - (24 if speaker else 0)
        pdf.multi_cell(remaining_w, 5, text)
    if len(transcript) > 50:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*GRAY)
        pdf.cell(0, 5, "Showing first 50 segments...", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Footer
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GRAY)
    pdf.cell(
        0, 5,
        "Generated by InsightAI  |  Powered by Simpleem-style 55/38/7 Signal Analysis",
        align="C",
    )

    return bytes(pdf.output())


def _section_heading(pdf: FPDF, title: str, color: tuple):
    """Render a section heading."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _rows_to_csv(rows, columns: list) -> str:
    """Convert aiosqlite rows to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[col] for col in columns])
    return output.getvalue()


def _format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"
