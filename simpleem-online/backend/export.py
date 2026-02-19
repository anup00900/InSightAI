"""PDF and CSV export for meeting analysis reports.

- PDF: fpdf2 (pure-Python, no system dependencies)
- CSV: ZIP file with engagement.csv, transcript.csv, participants.csv
"""

import csv
import io
import json
import logging
import os
import zipfile
from typing import Optional

import aiosqlite
from fpdf import FPDF

from .models import DB_PATH

logger = logging.getLogger(__name__)


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

            # Summary
            cursor = await db.execute(
                "SELECT * FROM summaries WHERE video_id = ?", (video_id,)
            )
            row = await cursor.fetchone()
            if row:
                summary_str = f"Summary\n{row['summary']}\n\nKey Topics\n{row['key_topics']}\n\nOverall Sentiment\n{row['overall_sentiment']}"
                zf.writestr("summary.txt", summary_str)

        return zip_buffer.getvalue()


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

    return _build_pdf(video, participants, summary, flags, transcript)


def _build_pdf(video, participants, summary, flags, transcript) -> bytes:
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
