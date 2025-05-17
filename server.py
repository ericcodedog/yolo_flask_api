from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2
from ultralytics import YOLO
import time
import gc

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("weights/yolov10s.pt")
model.fuse()

inference_counter = 0
CLEAR_INTERVAL = 50  # 每 50 次推論清一次 GPU 記憶體 2

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    global inference_counter
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    try:
        image_bytes = request.files['image'].read()
        img = preprocess_image(image_bytes)

        start = time.time()
        results = model.predict(source=img, device=device, verbose=False)
        elapsed = time.time() - start

        detections = []
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class': int(cls)
            })

        inference_counter += 1
        if inference_counter >= CLEAR_INTERVAL:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            inference_counter = 0

        return jsonify({'detections': detections, 'inference_time': elapsed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/ping", methods=["POST"])
def ping():
    start = time.time()
    # 模擬一下收到資料（但不做任何處理）
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    _ = request.files['image'].read()  # 讀取資料但不處理

    elapsed = time.time() - start
    return jsonify({'message': 'pong', 'elapsed_time': elapsed})


@app.route("/")
def home():
    return "YOLOv10 Flask API is running"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
