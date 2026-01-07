from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

model = tf.keras.models.load_model("mnist_ann_from_files.h5")

def preprocess_digit(img):
    h, w = img.shape
    size = max(h, w)

    canvas = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img

    img = cv2.resize(canvas, (28, 28))
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 784)
    return img

def predict_number(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 15 and h > 15:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    digits = []
    for x, y, w, h in boxes:
        digit_img = thresh[y:y+h, x:x+w]
        digit_img = preprocess_digit(digit_img)
        pred = np.argmax(model.predict(digit_img, verbose=0))
        digits.append(str(pred))

    return "".join(digits)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)
        result = predict_number(path)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
