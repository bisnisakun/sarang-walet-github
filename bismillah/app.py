from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import os

# =========================================
# KONFIGURASI
# =========================================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# otomatis buat folder uploads jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================================
# HOME
# =========================================
@app.route("/")
def index():

    files = []

    for filename in os.listdir(UPLOAD_FOLDER):

        file_path = os.path.join(UPLOAD_FOLDER, filename)

        if os.path.isfile(file_path):

            upload_time = datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).strftime("%d %B %Y %H:%M")

            files.append({
                "filename": filename,
                "time": upload_time
            })

    # urut terbaru
    files = sorted(files, key=lambda x: x["time"], reverse=True)

    return render_template(
        "index.html",
        files=files
    )

# =========================================
# UPLOAD FOTO
# =========================================
@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:
        return {
            "status": "error",
            "message": "Tidak ada file"
        }, 400

    file = request.files["image"]

    if file.filename == "":
        return {
            "status": "error",
            "message": "Nama file kosong"
        }, 400

    filename = secure_filename(file.filename)

    save_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(save_path)

    return {
        "status": "success",
        "message": "Upload berhasil",
        "filename": filename
    }

# =========================================
# DOWNLOAD FILE
# =========================================
@app.route("/download/<filename>")
def download_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
        as_attachment=True
    )

# =========================================
# DELETE FILE
# =========================================
@app.route("/delete/<filename>")
def delete_file(filename):

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if os.path.exists(file_path):
        os.remove(file_path)

    return redirect(url_for("index"))

# =========================================
# PREVIEW GAMBAR
# =========================================
@app.route("/view/<filename>")
def view_image(filename):

    image_url = url_for(
        "static",
        filename=f"uploads/{filename}"
    )

    return f"""
    <html>
    <head>
        <title>{filename}</title>

        <style>
            body{{
                background:#0f172a;
                display:flex;
                justify-content:center;
                align-items:center;
                height:100vh;
                margin:0;
                font-family:Arial;
            }}

            img{{
                max-width:90%;
                max-height:90%;
                border-radius:20px;
                box-shadow:0 0 30px rgba(0,0,0,0.5);
            }}
        </style>

    </head>

    <body>

        <img src="{image_url}">

    </body>
    </html>
    """

# =========================================
# VERCEL
# =========================================
app = app

# =========================================
# RUN
# =========================================
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )