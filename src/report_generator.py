"""
Report Generator Module for Intelligent Dataset Diagnosis and Quality Analysis System.
Generates comprehensive PDF reports, JSON audit exports, and CSV/Excel downloads.
"""

from datetime import datetime
import io
import json
from typing import Any, Dict, List, Optional
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class ReportGenerator:
    """Generates production-grade PDF, JSON, CSV, and Excel reports."""

    def generate_json_report(
        self,
        diagnosis: Dict[str, Any],
        cleaning_history: Optional[List[Dict[str, Any]]] = None,
        ai_explanation: Optional[str] = None
    ) -> str:
        """
        Generate complete structured JSON report.
        """
        report_data = {
            "metadata": {
                "system": "Intelligent Dataset Diagnosis and Quality Analysis System",
                "version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
                "dataset_name": diagnosis.get("dataset_name", "Dataset")
            },
            "quality_scores": diagnosis.get("quality_scores", {}),
            "overview": diagnosis.get("profile", {}).get("overview", {}),
            "missing_values_analysis": diagnosis.get("missing_analysis", {}),
            "duplicate_analysis": diagnosis.get("duplicate_analysis", {}),
            "outlier_analysis": diagnosis.get("outlier_analysis", {}),
            "data_type_analysis": diagnosis.get("dtype_analysis", {}),
            "detected_issues": diagnosis.get("detected_issues", []),
            "cleaning_history": cleaning_history or [],
            "ai_insights": ai_explanation or ""
        }
        return json.dumps(report_data, indent=2, default=str)

    def generate_pdf_report(
        self,
        diagnosis: Dict[str, Any],
        cleaning_history: Optional[List[Dict[str, Any]]] = None,
        ai_explanation: Optional[str] = None
    ) -> bytes:
        """
        Generate a professional PDF Data Quality Audit Report.
        
        Returns:
            bytes containing the PDF document.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0B0F19"),
            fontName="Helvetica-Bold",
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=15
        )
        h2_style = ParagraphStyle(
            "SectionH2",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1E293B"),
            fontName="Helvetica-Bold",
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            "BodyTextCustom",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155")
        )
        small_style = ParagraphStyle(
            "SmallText",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748B")
        )

        story = []

        # Header Title
        story.append(Paragraph("Intelligent Dataset Diagnosis & Quality Audit Report", title_style))
        gen_time = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
        dataset_name = diagnosis.get("dataset_name", "Uploaded Dataset")
        story.append(Paragraph(f"Dataset: <b>{dataset_name}</b> | Generated: {gen_time}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3B82F6"), spaceAfter=15))

        # 1. Executive Summary & Quality Score
        story.append(Paragraph("1. Executive Summary & Quality Score", h2_style))
        scores = diagnosis.get("quality_scores", {})
        overall_score = scores.get("overall_score", 0.0)
        grade = scores.get("grade", "N/A")
        status = scores.get("status", "N/A")

        score_table_data = [
            [
                Paragraph("<b>Overall Quality Score</b>", body_style),
                Paragraph(f"<b><font size=14 color='#2563EB'>{overall_score} / 100</font></b>", body_style),
                Paragraph(f"<b>Grade:</b> {grade}", body_style),
                Paragraph(f"<b>Readiness:</b> {status}", body_style)
            ]
        ]
        score_table = Table(score_table_data, colWidths=[1.8 * inch, 1.3 * inch, 1.4 * inch, 2.7 * inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 10))

        # Dimension Breakdown Table
        dims = scores.get("dimensions", {})
        dim_table_data = [
            ["Dimension", "Score", "Target", "Status"],
            ["Completeness", f"{dims.get('completeness', {}).get('score', 0)}%", ">= 95%", "Pass" if dims.get('completeness', {}).get('score', 0) >= 95 else "Flagged"],
            ["Uniqueness", f"{dims.get('uniqueness', {}).get('score', 0)}%", "100%", "Pass" if dims.get('uniqueness', {}).get('score', 0) == 100 else "Flagged"],
            ["Validity", f"{dims.get('validity', {}).get('score', 0)}%", ">= 90%", "Pass" if dims.get('validity', {}).get('score', 0) >= 90 else "Flagged"],
            ["Consistency", f"{dims.get('consistency', {}).get('score', 0)}%", ">= 90%", "Pass" if dims.get('consistency', {}).get('score', 0) >= 90 else "Flagged"],
        ]
        dim_table = Table(dim_table_data, colWidths=[2.2 * inch, 1.6 * inch, 1.6 * inch, 1.8 * inch])
        dim_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])
        ]))
        story.append(dim_table)
        story.append(Spacer(1, 14))

        # 2. Dataset Overview
        story.append(Paragraph("2. Dataset Overview & Dimensions", h2_style))
        profile = diagnosis.get("profile", {}).get("overview", {})
        overview_data = [
            ["Metric", "Value", "Metric", "Value"],
            ["Total Rows", str(profile.get("total_rows", 0)), "Total Columns", str(profile.get("total_columns", 0))],
            ["Total Data Cells", str(profile.get("total_cells", 0)), "Missing Cells", f"{profile.get('total_missing_cells', 0)} ({profile.get('missing_percentage', 0)}%)"],
            ["Duplicate Rows", f"{profile.get('duplicate_rows_count', 0)} ({profile.get('duplicate_percentage', 0)}%)", "Numerical Columns", str(profile.get("numerical_columns_count", 0))],
            ["Categorical Columns", str(profile.get("categorical_columns_count", 0)), "Datetime Columns", str(profile.get("datetime_columns_count", 0))]
        ]
        overview_table = Table(overview_data, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")])
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 14))

        # 3. Missing Values & Outlier Findings
        story.append(Paragraph("3. Detailed Quality Findings", h2_style))
        missing_cols = diagnosis.get("missing_analysis", {}).get("column_breakdown", [])
        if missing_cols:
            miss_rows = [["Column Name", "Missing Count", "Missing %", "Severity Level"]]
            for mc in missing_cols[:6]:
                miss_rows.append([mc["column"], str(mc["missing_count"]), f"{mc['missing_percentage']}%", mc["severity"]])
            miss_table = Table(miss_rows, colWidths=[2.6 * inch, 1.5 * inch, 1.5 * inch, 1.6 * inch])
            miss_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#475569")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ('PADDING', (0, 0), (-1, -1), 4)
            ]))
            story.append(Paragraph("<b>Top Columns with Missing Values:</b>", body_style))
            story.append(Spacer(1, 3))
            story.append(miss_table)
            story.append(Spacer(1, 8))

        outliers = diagnosis.get("outlier_analysis", {})
        outlier_cols = outliers.get("columns_with_outliers", [])
        if outlier_cols:
            out_rows = [["Column Name", "Outliers", "Outlier %", "Allowed IQR Range"]]
            for oc in outlier_cols[:6]:
                rep = outliers["column_reports"][oc]
                out_rows.append([oc, str(rep["outlier_count"]), f"{rep['outlier_pct']}%", f"[{rep['lower_bound']} to {rep['upper_bound']} ]"])
            out_table = Table(out_rows, colWidths=[2.2 * inch, 1.2 * inch, 1.4 * inch, 2.4 * inch])
            out_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#475569")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ('PADDING', (0, 0), (-1, -1), 4)
            ]))
            story.append(Paragraph("<b>Statistical Outliers Detected (IQR 1.5x Method):</b>", body_style))
            story.append(Spacer(1, 3))
            story.append(out_table)
            story.append(Spacer(1, 10))

        # 4. Generative AI Recommendations
        if ai_explanation:
            story.append(PageBreak())
            story.append(Paragraph("4. Generative AI Diagnostic Insights & Action Plan", h2_style))
            # Clean markdown formatting for ReportLab paragraphs
            cleaned_text = ai_explanation.replace("### ", "<b>").replace("## ", "<b>").replace("# ", "<b>")
            for line in cleaned_text.split("\n"):
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 4))
                elif line.startswith("- "):
                    story.append(Paragraph(f"• {line[2:]}", body_style))
                else:
                    story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 12))

        # 5. Cleaning Audit Trail
        if cleaning_history:
            story.append(Paragraph("5. Data Cleaning Studio Audit Trail", h2_style))
            clean_rows = [["Operation", "Details", "Rows Before", "Rows After", "Net Delta"]]
            for ch in cleaning_history:
                clean_rows.append([
                    ch.get("action", "Operation"),
                    Paragraph(str(ch.get("details", "")), small_style),
                    str(ch.get("rows_before", "-")),
                    str(ch.get("rows_after", "-")),
                    str(ch.get("rows_delta", 0))
                ])
            clean_table = Table(clean_rows, colWidths=[1.8 * inch, 2.8 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch])
            clean_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(clean_table)
            story.append(Spacer(1, 14))

        # Disclaimer / Limitations
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
        story.append(Paragraph(
            "<b>Methodology & Limitations Disclaimer:</b> Quality scores are computed deterministically across four fundamental "
            "dimensions (Completeness, Uniqueness, Validity, Consistency). This audit provides statistical and structural diagnosis "
            "and should be coupled with domain-specific governance.",
            small_style
        ))

        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    @staticmethod
    def export_csv(df: pd.DataFrame) -> bytes:
        """Export DataFrame as UTF-8 CSV bytes."""
        return df.to_csv(index=False).encode("utf-8")

    @staticmethod
    def export_excel(df: pd.DataFrame) -> bytes:
        """Export DataFrame as Excel .xlsx bytes."""
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
        buffer.seek(0)
        return buffer.getvalue()

