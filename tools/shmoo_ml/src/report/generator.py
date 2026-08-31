"""
ReportGenerator
---------------
Generates high-quality PDF reports using ReportLab.
Features executive summary, methodology, embedded high-res Shmoo plot,
numerical metrics table, yield-by-voltage table, multi-device ranking table,
failure breakdown, and recommendations.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from pathlib import Path

DARK_BLUE  = colors.HexColor('#1B2A4A')
MID_BLUE   = colors.HexColor('#2C4A7C')
ACCENT     = colors.HexColor('#3498DB')
GREEN      = colors.HexColor('#27AE60')
RED        = colors.HexColor('#C0392B')
ORANGE     = colors.HexColor('#E67E22')
LIGHT_GREY = colors.HexColor('#F4F6F9')
BORDER     = colors.HexColor('#BDC3C7')


def _styles():
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('Title', parent=base['Normal'], fontSize=20, fontName='Helvetica-Bold', textColor=DARK_BLUE, spaceAfter=4),
        'subtitle': ParagraphStyle('Subtitle', parent=base['Normal'], fontSize=10, fontName='Helvetica', textColor=MID_BLUE, spaceAfter=10),
        'h1': ParagraphStyle('H1', parent=base['Normal'], fontSize=13, fontName='Helvetica-Bold', textColor=DARK_BLUE, spaceBefore=12, spaceAfter=6, leading=16),
        'h2': ParagraphStyle('H2', parent=base['Normal'], fontSize=10.5, fontName='Helvetica-Bold', textColor=MID_BLUE, spaceBefore=8, spaceAfter=4),
        'body': ParagraphStyle('Body', parent=base['Normal'], fontSize=9, fontName='Helvetica', leading=13.5, alignment=TA_JUSTIFY),
        'caption': ParagraphStyle('Caption', parent=base['Normal'], fontSize=8, fontName='Helvetica-Oblique', textColor=colors.grey, alignment=TA_CENTER),
    }


class ReportGenerator:
    def generate(self, results, meta, narrative, plot_path, output_path, results_by_die=None, test_methodology='MBIST'):
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=1.8*cm, rightMargin=1.8*cm,
            topMargin=2.0*cm, bottomMargin=2.0*cm,
            title=f'{test_methodology} SHMOO Analysis Report', author='Shmoo ML System'
        )

        S = _styles()
        story = []
        W = doc.width

        # Header
        story.append(Paragraph(f'{test_methodology} SHMOO Characterization & Analysis Report', S['title']))
        story.append(Paragraph(f'<i>Die-Level VDD/Frequency Characterization — {test_methodology} Binning & Screening Recommendation</i>', S['subtitle']))
        
        # Metadata Bar
        meta_text = f"Source: {meta.get('lot_id', 'Lot_A001')}, Wafer {meta.get('wafer_id', 'W001')}, Devices: {meta.get('n_dies', 1)}  |  Prepared: {datetime.now().strftime('%B %d, %Y')}  |  Pass Rate: {meta.get('pass_rate', 0)*100:.1f}%"
        story.append(Paragraph(meta_text, S['caption']))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width='100%', thickness=1.5, color=DARK_BLUE))
        story.append(Spacer(1, 8))

        # 1. Executive Summary
        story.append(Paragraph('1. Executive Summary — AI Recommendation', S['h1']))
        for para in narrative.split('\n\n'):
            if para.strip():
                story.append(Paragraph(para.strip(), S['body']))
                story.append(Spacer(1, 4))
        story.append(Spacer(1, 6))

        # 2. Methodology
        story.append(Paragraph('2. Methodology & Model Performance', S['h1']))
        story.append(Paragraph(
            f"Dataset contains {meta['n_points']:,} test points across VDD [{meta['vdd_range'][0]:.2f} V - {meta['vdd_range'][1]:.2f} V] "
            f"and Frequency [{meta['freq_range'][0]:.2f} GHz - {meta['freq_range'][1]:.2f} GHz]. "
            f"Model: Gradient Boosting Classifier (5-fold Stratified CV accuracy: {results.cv_accuracy*100:.2f}%) "
            f"+ RANSAC Linear Boundary Extractor (R² = {results.boundary_r2:.4f}).", S['body']
        ))
        story.append(Spacer(1, 6))

        # 3. Shmoo Plot
        story.append(Paragraph('3. Interactive SHMOO Characterization Plot', S['h1']))
        if Path(plot_path).exists():
            img = Image(plot_path, width=W, height=W*0.6)
            story.append(img)
            story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"Figure 1 — Shmoo Plot. Linear boundary: Fmax(GHz) ≈ {results.boundary_slope:.3f} × VDD(V) "
            f"{'+' if results.boundary_intercept>=0 else ''}{results.boundary_intercept:.3f}", S['caption']
        ))
        story.append(Spacer(1, 6))

        # 3.1 Shmoo Plot Classification & Diagnostic Analysis
        shmoo_type = getattr(results, 'shmoo_plot_type', 'Normal Shmoo')
        shmoo_desc = getattr(results, 'shmoo_type_description', '')
        if shmoo_type:
            story.append(Paragraph('3.1 Automated Shmoo Classification & Diagnostic Analysis', S['h2']))
            diag_text = f"<b>Identified Shmoo Pattern:</b> {shmoo_type}<br/><b>Diagnostic Analysis:</b> {shmoo_desc}"
            story.append(Paragraph(diag_text, S['body']))
            story.append(Spacer(1, 8))

        # 4. Numerical Analysis
        story.append(Paragraph('4. Numerical Metrics & Margin Analysis', S['h1']))
        story.append(self._metrics_table(results, S, W))
        story.append(Spacer(1, 8))

        # 4.1 Yield Table
        story.append(Paragraph('4.1 Pass/Fail Boundary & Yield by Voltage', S['h2']))
        story.append(self._yield_table(results, S, W))
        story.append(Spacer(1, 8))

        # 4.2 Multi-Device Performance Ranking (If dataset has multiple devices)
        if getattr(results, 'is_multi_die', False) and getattr(results, 'die_rankings', None):
            story.append(Paragraph('4.2 Multi-Device Binning Performance Ranking', S['h2']))
            story.append(self._multi_die_table(results, S, W))
            story.append(Spacer(1, 8))

        # 5. Failure Mode Analysis
        story.append(Paragraph('5. Failure Mode Analysis', S['h1']))
        story.append(self._failure_table(results, S, W))
        story.append(Spacer(1, 8))

        critical_patterns = getattr(results, 'critical_fault_patterns', getattr(results, 'timing_fail_patterns', []))
        if critical_patterns:
            story.append(Paragraph('5.1 Top Critical Fault Patterns', S['h2']))
            story.append(self._pattern_table(results, S, W))
            story.append(Spacer(1, 8))

        # 6. Screening Recommendations
        story.append(Paragraph('6. Production Binning & Guardband Recommendation', S['h1']))
        rec_text = (
            f"<b>Primary Operating Point:</b> Set VDD ≥ {results.recommended_vdd:.3f} V and Frequency ≤ {results.recommended_freq:.3f} GHz. "
            f"This provides a voltage margin of {results.voltage_margin_v*1000:.0f} mV and a frequency margin of {results.freq_margin_ghz*1000:.0f} MHz. "
        )
        if getattr(results, 'is_multi_die', False) and getattr(results, 'low_performer', None):
            high = results.high_performer
            low = results.low_performer
            rec_text += (
                f"<b>Multi-Device Strategy:</b> High Performer device <b>{high['die_id']}</b> achieves Fmax of {high['fmax_at_nom']:.2f} GHz. "
                f"Population overall recommendation is bounded by Low Performer device <b>{low['die_id']}</b> (Fmax of {low['fmax_at_nom']:.2f} GHz) "
                f"to guarantee 100% binning yield across all devices."
            )
        else:
            rec_text += (
                f"<b>Screening Strategy:</b> Employ a two-tier screening process: (1) Standard VDD/Freq sweep for FREQ_MARGIN fails, "
                f"and (2) Targeted screening for critical functional/hard defects."
            )
        story.append(Paragraph(rec_text, S['body']))
        story.append(Spacer(1, 10))

        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)

    def _metrics_table(self, results, S, W):
        data = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['ML Test Accuracy', f'{results.accuracy*100:.2f}%', '5-Fold CV Accuracy', f'{results.cv_accuracy*100:.2f}% ± {results.cv_std*100:.2f}%'],
            ['Boundary Slope', f'{results.boundary_slope:.4f} GHz/V', 'Boundary Intercept', f'{results.boundary_intercept:.4f} GHz'],
            ['Boundary Fit (R²)', f'{results.boundary_r2:.4f}', 'Recommended VDD', f'{results.recommended_vdd:.3f} V'],
            ['Recommended Freq', f'{results.recommended_freq:.3f} GHz', 'Voltage Margin', f'{results.voltage_margin_v*1000:.0f} mV'],
            ['Frequency Margin', f'{results.freq_margin_ghz*1000:.0f} MHz', 'Total PASS / FAIL', f'{results.n_pass} / {results.n_fail}'],
        ]
        tbl = Table(data, colWidths=[W*0.28, W*0.22, W*0.28, W*0.22])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), DARK_BLUE),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GREY]),
            ('GRID', (0,0), (-1,-1), 0.4, BORDER),
            ('ALIGN', (1,0), (1,-1), 'CENTER'),
            ('ALIGN', (3,0), (3,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        return tbl

    def _yield_table(self, results, S, W):
        header = [['VDD (V)', 'Fmax @ Boundary (GHz)', 'Yield (Pass %)']]
        rows = []
        for vdd in sorted(results.yield_by_vdd.keys()):
            fmax = results.fmax_by_vdd.get(vdd, '—')
            fmax_str = f'{fmax:.2f}' if isinstance(fmax, float) else '—'
            yield_pct = results.yield_by_vdd[vdd] * 100
            rows.append([f'{vdd:.2f}', fmax_str, f'{yield_pct:.0f}%'])
        tbl = Table(header + rows, colWidths=[W*0.25, W*0.40, W*0.35])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), MID_BLUE),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GREY]),
            ('GRID', (0,0), (-1,-1), 0.4, BORDER),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 3),
        ]))
        return tbl

    def _multi_die_table(self, results, S, W):
        data = [['Device ID', 'Performance Category', 'Fmax @ Nom VDD', 'Yield (%)', 'Rec. Freq (GHz)']]
        rankings = getattr(results, 'die_rankings', [])
        high = getattr(results, 'high_performer', None)
        low = getattr(results, 'low_performer', None)

        for d in rankings:
            die_id = d['die_id']
            if high and die_id == high['die_id']:
                cat = 'High Performer (Fastest)'
            elif low and die_id == low['die_id']:
                cat = 'Low Performer (Corner)'
            else:
                cat = 'Nominal Device'

            data.append([
                die_id,
                cat,
                f"{d['fmax_at_nom']:.2f} GHz",
                f"{d['pass_rate']*100:.1f}%",
                f"{d['recommended_freq']:.2f} GHz"
            ])

        tbl = Table(data, colWidths=[W*0.18, W*0.34, W*0.18, W*0.14, W*0.16])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), DARK_BLUE),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GREY]),
            ('GRID', (0,0), (-1,-1), 0.4, BORDER),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 3),
        ]))
        return tbl

    def _failure_table(self, results, S, W):
        dist = results.failure_code_dist
        total_fails = results.n_fail or 1
        data = [['Failure Code', 'Count', '% of Fails', 'Description']]
        descs = {
            'FREQ_MARGIN':                'Voltage/frequency-limited — above Fmax at given VDD',
            'TIMING':                      'Pattern-specific critical path fail (scan)',
            'STUCK_AT':                    'Memory cell stuck at fixed logic value — hard defect',
            'COUPLING_FAULT':              'Adjacent-cell coupling fault — hard defect',
            'RETENTION_FAULT':             'Cell fails to retain data — hard defect',
            'ADDRESS_DECODE_FAULT':        'Address decoder fault — hard defect',
            'MARGINALITY_IR_DROP':         'Marginality Shmoo: Supply rail droop & clock skew sensitivity',
            'POWER_IR_DROP_FAIL':          'Power Issue: Excessive dynamic switching current & IR drop on supply rail',
            'HOLD_TIME_VIOLATION':        'Hold Time Issue: Fast-path logic race condition & clock skew at high VDD',
            'MARGINALITY_VDDL_FAIL':       'Marginality Issue: Low voltage rail (VDDL) setup timing failure',
            'MARGINALITY_VDDH_FAIL':       'Marginality Issue: High voltage rail (VDDH) PLL jitter/droop failure',
            'NA':                          'Unclassified failure',
        }
        for code, count in sorted(dist.items(), key=lambda x: -x[1]):
            if code == 'NA' and count == 0: continue
            desc = descs.get(code, 'Functional defect / failure code')
            data.append([code, str(count), f'{count/total_fails*100:.1f}%', desc])
        tbl = Table(data, colWidths=[W*0.25, W*0.12, W*0.15, W*0.48])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), DARK_BLUE),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GREY]),
            ('GRID', (0,0), (-1,-1), 0.4, BORDER),
            ('ALIGN', (1,0), (2,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 3),
        ]))
        return tbl

    def _pattern_table(self, results, S, W):
        critical_patterns = getattr(results, 'critical_fault_patterns', getattr(results, 'timing_fail_patterns', []))
        data = [['Source (Pattern / March Alg & Instance)', 'Fault Type', 'Fail Count']]
        for p in critical_patterns:
            source = p.get('source', p.get('pattern', 'N/A'))
            fault_type = p.get('fault_type', 'TIMING')
            data.append([source, fault_type, str(p['fail_count'])])
        tbl = Table(data, colWidths=[W*0.48, W*0.32, W*0.20])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), ORANGE),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GREY]),
            ('GRID', (0,0), (-1,-1), 0.4, BORDER),
            ('ALIGN', (1,0), (2,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 3),
        ]))
        return tbl

    @staticmethod
    def _header_footer(canvas, doc):
        canvas.saveState()
        W, H = A4
        canvas.setFillColor(DARK_BLUE)
        canvas.rect(0, H-1.2*cm, W, 1.2*cm, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont('Helvetica-Bold', 9)
        canvas.drawString(1.8*cm, H-0.8*cm, 'SHMOO Analysis Report — Confidential')
        canvas.setFont('Helvetica', 8)
        canvas.drawRightString(W-1.8*cm, H-0.8*cm, datetime.now().strftime('%Y-%m-%d'))
        
        canvas.setFillColor(LIGHT_GREY)
        canvas.rect(0, 0, W, 0.8*cm, fill=1, stroke=0)
        canvas.setFillColor(colors.grey)
        canvas.setFont('Helvetica', 7.5)
        canvas.drawString(1.8*cm, 0.3*cm, 'Generated locally by Shmoo ML System')
        canvas.drawRightString(W-1.8*cm, 0.3*cm, f'Page {doc.page}')
        canvas.restoreState()
