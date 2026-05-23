"""
app.py
Flask REST API — Dog vs Cat Image Classification using fine-tuned ResNet18

Endpoints:
  GET  /              → health check
  POST /predict       → single image prediction (upload file)
  POST /batch_predict → batch prediction (upload up to 4 images)
"""

import io
import os
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import models, transforms
import torch.nn as nn

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "./model_output/resnet18_dogcat.pth"
IMAGE_SIZE  = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID2LABEL    = {0: "cat", 1: "dog"}
ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# LOAD MODEL ONCE AT STARTUP
# ─────────────────────────────────────────────
def load_model(path: str):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


print(f"[startup] Loading model from '{MODEL_PATH}' on {DEVICE} ...")
model = load_model(MODEL_PATH)
print("[startup] Model loaded successfully ✅")

app = Flask(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def predict_image_bytes(image_bytes: bytes) -> dict:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred   = int(np.argmax(probs))

    return {
        "label":      ID2LABEL[pred],
        "label_id":   pred,
        "confidence": round(float(max(probs)), 4),
        "scores": {
            "cat": round(float(probs[0]), 4),
            "dog": round(float(probs[1]), 4),
        },
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status":  "ok",
        "model":   "ResNet18 (fine-tuned on Dog vs Cat)",
        "task":    "Image Classification",
        "classes": ["cat", "dog"],
        "device":  str(DEVICE),
        "version": "1.0.0",
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single image prediction.
    Request: multipart/form-data with field 'image' (jpg/png file)
    Response:
        { "filename": "...", "label": "dog", "confidence": 0.97, "scores": {...} }
    """
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Use: {ALLOWED_EXT}"}), 400

    image_bytes = file.read()
    try:
        result = predict_image_bytes(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"filename": file.filename, **result})


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Batch image prediction (up to 4 images).
    Request: multipart/form-data with fields 'images' (multiple files)
    Response:
        { "count": 4, "results": [ { "filename": "...", "label": "...", ... }, ... ] }
    """
    files = request.files.getlist("images")

    if not files or len(files) == 0:
        return jsonify({"error": "No 'images' files in request."}), 400
    if len(files) > 4:
        return jsonify({"error": "Maximum 4 images per batch."}), 400

    results = []
    for file in files:
        if not allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "error": f"File type not allowed.",
            })
            continue
        try:
            image_bytes = file.read()
            result = predict_image_bytes(image_bytes)
            results.append({"filename": file.filename, **result})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return jsonify({"count": len(results), "results": results})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
