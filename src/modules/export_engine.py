# AetherForge v1.0 — src/modules/export_engine.py
# ─────────────────────────────────────────────────────────────────
# Converts chat sessions / individual messages to downloadable
# .md and .pdf documents.
#
# PDF rendering chain:
#   session messages → Markdown string → HTML → PDF (weasyprint)
#   Falls back to reportlab if weasyprint/cairo not installed.
#
# Chart support:
#   If a response contains VLM visual analysis with bullet-point
#   chart descriptions, the engine extracts data and embeds a
#   matplotlib chart image in the PDF.
# ─────────────────────────────────────────────────────────────────
from __future__ import annotations

import io
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.modules.session_store import SessionStore, StoredMessage

logger = logging.getLogger("aetherforge.export_engine")

# ── AetherForge CSS theme for PDF output ─────────────────────────
_PDF_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 11pt;
    line-height: 1.65;
    color: #1a1a2e;
    padding: 36px 54px;
    max-width: 750px;
    margin: 0 auto;
}

h1 { font-size: 20pt; color: #0f3460; border-bottom: 2px solid #e94560; padding-bottom: 8px; margin-bottom: 16px; }
h2 { font-size: 14pt; color: #16213e; margin: 20px 0 8px; }
h3 { font-size: 11pt; color: #1a1a2e; margin: 12px 0 4px; }

p { margin-bottom: 10px; }
ul, ol { margin: 8px 0 10px 24px; }
li { margin-bottom: 4px; }

code { font-family: 'Courier New', monospace; background: #f0f2f5; padding: 1px 4px; border-radius: 3px; font-size: 10pt; }
pre { background: #f0f2f5; padding: 12px; border-radius: 6px; overflow-x: auto; margin: 10px 0; }
pre code { background: none; padding: 0; }

blockquote { border-left: 4px solid #e94560; padding-left: 14px; color: #444; margin: 12px 0; }

.message-block { margin-bottom: 20px; border-radius: 8px; padding: 14px 18px; }
.user-block   { background: #eef2ff; border-left: 4px solid #4f86e8; }
.ai-block     { background: #f8f9fa; border-left: 4px solid #e94560; }

.role-label {
    font-size: 9pt; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #666; margin-bottom: 6px;
}

.meta { font-size: 9pt; color: #888; margin-top: 6px; text-align: right; }

.header-bar {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 3px solid #0f3460; padding-bottom: 12px; margin-bottom: 28px;
}
.brand { font-size: 18pt; font-weight: 700; color: #0f3460; }
.brand span { color: #e94560; }
.export-date { font-size: 9pt; color: #888; }

img.chart { max-width: 100%; margin: 10px 0; border-radius: 6px; }

table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10pt; }
th { background: #0f3460; color: white; padding: 6px 10px; text-align: left; }
td { padding: 5px 10px; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) td { background: #f8f9fa; }
"""


class ExportEngine:
    """
    Converts sessions/messages to .md and .pdf artifacts.

    Usage:
        engine = ExportEngine(session_store)
        md_str  = engine.session_to_markdown(session_id)
        pdf_bytes = engine.session_to_pdf(session_id)
    """

    def __init__(self, session_store: "SessionStore") -> None:
        self.store = session_store

    # ─────────────────────────────────────────────────────────────
    # PUBLIC: Markdown
    # ─────────────────────────────────────────────────────────────

    def session_to_markdown(self, session_id: str) -> str:
        """Export a full session as GitHub-flavored Markdown."""
        sessions = {s.id: s for s in self.store.list_sessions()}
        session = sessions.get(session_id)
        title = session.title if session else "AetherForge Session"
        messages = self.store.get_messages(session_id)

        lines = [
            f"# {title}",
            f"",
            f"*Exported from AetherForge · {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            f"",
            "---",
            "",
        ]
        for msg in messages:
            if msg.role == "system":
                continue
            role_label = "**You**" if msg.role == "user" else "**Æ (AetherForge AI)**"
            ts = datetime.fromtimestamp(msg.ts).strftime("%H:%M")
            lines += [
                f"{role_label} · *{ts}*",
                "",
                msg.content,
                "",
                "---",
                "",
            ]
        return "\n".join(lines)

    def message_to_markdown(self, session_id: str, message_id: str) -> str:
        """Export a single AI response as Markdown."""
        messages = self.store.get_messages(session_id)
        # find the user question just before this message
        question = ""
        answer = ""
        for i, m in enumerate(messages):
            if m.id == message_id:
                answer = m.content
                if i > 0 and messages[i - 1].role == "user":
                    question = messages[i - 1].content
                break

        lines = [
            "# AetherForge — Response Export",
            f"",
            f"*Exported · {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "---",
            "",
        ]
        if question:
            lines += [f"**Question:** {question}", ""]
        lines += [
            "**Answer:**",
            "",
            answer,
            "",
            "---",
            "",
            "*Generated by AetherForge RAGForge CognitiveRAG*",
        ]
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    # PUBLIC: PDF
    # ─────────────────────────────────────────────────────────────

    def session_to_pdf(self, session_id: str) -> bytes:
        """Export a full session as PDF. Returns raw bytes."""
        md = self.session_to_markdown(session_id)
        return self._markdown_to_pdf(md)

    def message_to_pdf(self, session_id: str, message_id: str) -> bytes:
        """Export a single AI response as PDF. Returns raw bytes."""
        md = self.message_to_markdown(session_id, message_id)
        return self._markdown_to_pdf(md)

    # ─────────────────────────────────────────────────────────────
    # INTERNAL: MD → PDF
    # ─────────────────────────────────────────────────────────────

    def _markdown_to_pdf(self, md_text: str) -> bytes:
        """Convert markdown string to PDF bytes."""
        html = self._md_to_html(md_text)
        return self._html_to_pdf(html)

    def _md_to_html(self, md_text: str) -> str:
        """Convert markdown to a styled HTML document."""
        try:
            import markdown as md_lib
            body_html = md_lib.markdown(
                md_text,
                extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
            )
            
            # Convert file:// images to base64 for WeasyPrint/ReportLab stability
            import base64
            def _replace_img_with_base64(match):
                img_path = match.group(1).replace("file://", "")
                path = Path(img_path)
                if path.exists():
                    ext = path.suffix.lower().replace(".", "")
                    with open(path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                        return f'src="data:image/{ext};base64,{b64}"'
                return match.group(0)

            body_html = re.sub(r'src="file://([^"]+)"', _replace_img_with_base64, body_html)
        except ImportError:
            # Fallback: basic regex conversion
            body_html = self._basic_md_to_html(md_text)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{_PDF_CSS}</style>
</head>
<body>
<div class="header-bar">
  <div class="brand">Æther<span>Forge</span></div>
  <div class="export-date">{now_str}</div>
</div>
{body_html}
</body>
</html>"""

    def _html_to_pdf(self, html: str) -> bytes:
        """Render HTML to PDF bytes. Tries weasyprint then reportlab."""
        # ── Try weasyprint (best quality) ────────────────────────
        try:
            from weasyprint import HTML
            buf = io.BytesIO()
            HTML(string=html).write_pdf(buf)
            buf.seek(0)
            logger.info("PDF generated via weasyprint")
            return buf.read()
        except ImportError:
            logger.info("weasyprint not available — falling back to reportlab")
        except Exception as e:
            logger.warning("weasyprint failed: %s — falling back to reportlab", e)

        # ── Fallback: reportlab (pure Python) ────────────────────
        return self._reportlab_fallback(html)

    def _reportlab_fallback(self, html: str) -> bytes:
        """
        Minimal PDF via reportlab. Strips HTML tags and renders plain text
        in a clean layout. Installed via: pip install reportlab
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, HRFlowable
            )
            from reportlab.lib.enums import TA_LEFT

            # Strip HTML tags for plain-text reportlab render
            plain = re.sub(r"<[^>]+>", "", html)
            plain = plain.replace("&nbsp;", " ").replace("&amp;", "&")
            plain = plain.replace("&lt;", "<").replace("&gt;", ">")

            buf = io.BytesIO()
            doc = SimpleDocTemplate(
                buf, pagesize=A4,
                leftMargin=2*cm, rightMargin=2*cm,
                topMargin=2*cm, bottomMargin=2*cm
            )
            styles = getSampleStyleSheet()
            normal = styles["Normal"]
            normal.fontName = "Helvetica"
            normal.fontSize = 10
            normal.leading = 14

            heading = ParagraphStyle(
                "AEHeading", parent=styles["Heading1"],
                fontSize=14, textColor=colors.HexColor("#0f3460"),
                spaceAfter=8,
            )

            story = []
            story.append(Paragraph("AetherForge — Session Export", heading))
            story.append(HRFlowable(width="100%", thickness=1,
                                     color=colors.HexColor("#e94560")))
            story.append(Spacer(1, 12))

            for line in plain.split("\n"):
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 6))
                    continue
                if line.startswith("# "):
                    story.append(Paragraph(line[2:], heading))
                else:
                    try:
                        story.append(Paragraph(line, normal))
                    except Exception:
                        pass

            doc.build(story)
            buf.seek(0)
            logger.info("PDF generated via reportlab fallback")
            return buf.read()

        except ImportError:
            logger.error("Neither weasyprint nor reportlab available. "
                         "Install: pip install reportlab")
            raise RuntimeError(
                "No PDF renderer available. "
                "Run: pip install reportlab\n"
                "Or for full quality: brew install cairo pango && pip install weasyprint"
            )

    @staticmethod
    def _basic_md_to_html(md_text: str) -> str:
        """Bare-minimum markdown→HTML when the 'markdown' library isn't installed."""
        lines = md_text.split("\n")
        out = []
        for line in lines:
            line = line.rstrip()
            if line.startswith("### "):
                out.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("## "):
                out.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("# "):
                out.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("---"):
                out.append("<hr>")
            elif line.startswith("- "):
                out.append(f"<li>{line[2:]}</li>")
            elif re.match(r"^\d+\. ", line):
                out.append(f"<li>{re.sub(r'^\d+\. ', '', line)}</li>")
            elif line == "":
                out.append("<br>")
            else:
                # Inline: **bold**, *italic*, `code`
                line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
                line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
                line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
                out.append(f"<p>{line}</p>")
        return "\n".join(out)
