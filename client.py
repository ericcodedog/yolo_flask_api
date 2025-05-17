from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import requests
import time

app = Flask(__name__)
API_URL = "http://127.0.0.1:5000/predict"
IMAGE_FOLDER = "static/test_images"
RESULT_FOLDER = "static/results"
session = requests.Session()

os.makedirs(RESULT_FOLDER, exist_ok=True)

def run_inference_and_draw():
    image_filenames = [f for f in os.listdir(IMAGE_FOLDER)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    results = []
    total_pure_inference_time = 0  # 累積純推論時間（只統計 model.predict）
    total_request_time = 0         # 整體請求耗時（含網路）
    
    for filename in image_filenames:
        image_path = os.path.join(IMAGE_FOLDER, filename)
        with open(image_path, "rb") as f:
            files = {"image": f}
            start_request = time.time()
            response = session.post(API_URL, files=files)
            total_elapsed = time.time() - start_request  # 含網路等耗時

        if response.status_code != 200:
            continue

        result = response.json()
        detections = result["detections"]
        pure_infer_time = result["inference_time"]  # 純推論時間由 API 回傳

        # 畫框與存圖（另計時間）
        image = cv2.imread(image_path)
        draw_start = time.time()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["confidence"]
            cls = det["class"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{cls} ({conf:.2f})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        output_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(output_path, image)
        draw_time = time.time() - draw_start

        # 收集資訊
        results.append({
            "filename": filename,
            "inference_time": pure_infer_time,
            "draw_time": draw_time,
            "total_request_time": total_elapsed
        })

        total_pure_inference_time += pure_infer_time
        total_request_time += total_elapsed

    avg_fps = len(results) / total_request_time if total_request_time > 0 else 0
    return results, avg_fps

@app.route("/ping_test", methods=["POST"])
def ping_test():
    test_image_path = os.path.join(IMAGE_FOLDER, "test1.jpg")
    if not os.path.exists(test_image_path):
        return "測試圖片不存在，請放一張 test1.jpg 在 static/test_images 裡"

    with open(test_image_path, "rb") as f:
        files = {"image": f}
        start = time.time()
        response = requests.post(API_URL, files=files)
        elapsed = time.time() - start

    if response.status_code != 200:
        return f"API 回傳錯誤：{response.status_code} {response.text}"

    data = response.json()
    pure_infer_time = data.get("inference_time", None)

    return f"""
    <h2>API 響應測試結果</h2>
    <p>總請求時間（含網路與回傳）: {elapsed:.3f} 秒</p>
    <p>純推論時間（API回傳）: {pure_infer_time:.3f} 秒</p>
    <a href="/">返回</a>
    """




@app.route("/", methods=["GET"])
def index():
    image_filenames = os.listdir(IMAGE_FOLDER)
    image_filenames = [f for f in image_filenames if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return render_template("index.html", images=image_filenames, results=None, fps=None)

@app.route("/detect", methods=["POST"])
def detect():
    results, avg_fps = run_inference_and_draw()
    return render_template("index.html", images=None, results=results, fps=avg_fps)

if __name__ == "__main__":
    app.run(port=8001)
