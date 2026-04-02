import json
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def export_to_json(data, output_path="outputs/report.json"):
    """
    Exports prediction data to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    return output_path

def export_to_pdf(data, output_path="outputs/report.pdf"):
    """
    Exports prediction data to a PDF file using ReportLab.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    c = canvas.Canvas(output_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    
    # Title
    c.drawString(50, 750, "InstruNet AI - Analysis Report")
    
    # Summary
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, "Overall Detected Instruments:")
    
    y = 700
    for inst in data['summary']['detected_instruments']:
        text = f"- {inst['instrument']}: {inst['confidence']*100:.1f}% confidence"
        c.drawString(70, y, text)
        y -= 20
        
    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Segment Timeline:")
    c.setFont("Helvetica", 10)
    y -= 20
    
    for seg in data['timeline']:
        if y < 50: # Page break
            c.showPage()
            c.setFont("Helvetica", 10)
            y = 750
            
        time_text = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"
        
        # Find dominant instrument for this segment
        dominant_inst = max(seg['confidences'], key=seg['confidences'].get)
        max_conf = seg['confidences'][dominant_inst]
        
        details_text = f"Dominant: {dominant_inst} ({max_conf*100:.1f}%)"
        
        c.drawString(50, y, f"{time_text} {details_text}")
        y -= 15
        
    c.save()
    return output_path
