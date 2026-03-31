from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(result):
    doc = SimpleDocTemplate("outputs/report.pdf")
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("InstruNet Report", styles['Title']))
    elements.append(Spacer(1, 12))

    for k,v in result.items():
        elements.append(Paragraph(f"{k}: {v:.2f}", styles['Normal']))
        elements.append(Spacer(1, 8))

    doc.build(elements)