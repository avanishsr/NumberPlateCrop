import os
from flask import Flask, request, jsonify
import cv2
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load the YOLO model (ensure best.pt is in the same folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_numberPlate.pt")
model = YOLO(MODEL_PATH)

def process_image(image):
    # image: PIL Image in RGB mode
    image_np = np.array(image)  # RGB
    # convert RGB -> BGR for OpenCV / ultralytics if needed
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLO on the image
    results = model(image_bgr)  # ultralytics model inference

    # If any boxes detected, use the first one (change logic if needed)
    if len(results) > 0 and len(results[0].boxes) > 0:
        # xyxy[0] -> tensor of [x1,y1,x2,y2]
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        # clamp coordinates to image bounds
        h, w = image_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Crop using RGB numpy image
        cropped = image_np[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        pil_img = Image.fromarray(cropped)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_str

    return None

@app.route('/detect_number_plate', methods=['POST'])
def detect_number_plate():
    # Expecting form-data file field named 'file'
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded (form field 'file' missing)"}), 400

    file = request.files['file']

    # quick check for image content-type
    if not file.content_type.startswith("image/"):
        return jsonify({"error": "Uploaded file is not an image"}), 400

    try:
        image = Image.open(file).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    cropped_image_base64 = process_image(image)

    if cropped_image_base64:
        return jsonify({"cropped_image": cropped_image_base64})
    else:
        return jsonify({"error": "Could not detect number plate"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False for production; True if you want auto-reload locally
    app.run(host="0.0.0.0", port=port, debug=True)
