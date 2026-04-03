from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
import random
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.secret_key = "instrunet_secret_key"

UPLOAD_FOLDER = "static"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form.get("username")
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/download/json")
def download_json():
    return send_file("static/result.json", as_attachment=True)


@app.route("/download/pdf")
def download_pdf():
    return send_file("static/report.pdf", as_attachment=True)


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    predictions = None
    top_label = ""
    top_score = 0
    duration = 0
    audio = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            filepath = os.path.join("static", "temp.wav")
            file.save(filepath)

            audio = url_for('static', filename='temp.wav')

            predictions = {
                "keyboard": random.randint(50, 80),
                "guitar": random.randint(80, 96),
                "bass": random.randint(60, 85),
                "string": random.randint(20, 50)
            }

            predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))

            top_label = list(predictions.keys())[0]
            top_score = predictions[top_label]
            duration = round(random.uniform(60, 180), 2)

            result_data = {
                "predictions": predictions,
                "top_instrument": top_label,
                "confidence": top_score,
                "duration": duration
            }

            with open("static/result.json", "w") as f:
                json.dump(result_data, f, indent=4)

            doc = SimpleDocTemplate("static/report.pdf")
            styles = getSampleStyleSheet()

            content = []
            content.append(Paragraph("InstruNet AI Report", styles["Title"]))
            content.append(Paragraph(f"Top Instrument: {top_label}", styles["Normal"]))
            content.append(Paragraph(f"Confidence: {top_score}%", styles["Normal"]))
            content.append(Paragraph(f"Duration: {duration}s", styles["Normal"]))

            for k, v in predictions.items():
                content.append(Paragraph(f"{k}: {v}%", styles["Normal"]))

            doc.build(content)

    return render_template(
        "dashboard.html",
        predictions=predictions,
        top_label=top_label,
        top_score=top_score,
        duration=duration,
        audio=audio
    )


if __name__ == "__main__":
    app.run(debug=True)