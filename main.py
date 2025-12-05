# main.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from neuralic_3_0 import Neuralic

app = Flask(__name__, static_folder="static")
CORS(app)  # Allow frontend to access API

# Initialize AI
ai = Neuralic()
ai.load_state()

# Serve frontend
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    msg = data.get("message","").strip()
    if not msg:
        return jsonify({"reply":"Say something."})
    reply = ai.handle_input(msg)
    return jsonify({"reply": reply})

@app.route("/teach", methods=["POST"])
def teach():
    data = request.json
    msg = data.get("message","").strip()
    reply_text = data.get("reply","").strip()
    if not msg or not reply_text:
        return jsonify({"status":"error","message":"Provide both message and reply"})
    res = ai.teach(msg, reply_text)
    return jsonify({"status":"success","result":res})

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"status":"error","message":"No file uploaded"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status":"error","message":"No file selected"})
    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)
    res = ai.learn_file(filepath)
    return jsonify({"status":"success","result":res})

@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
