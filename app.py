from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os

app = Flask(__name__)

# ==========================
# Folder upload
# ==========================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================
# Load CNN
# ==========================
model = load_model("cnn_face_mask.h5")

# ==========================
# Load Haar Cascade
# ==========================
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Label kelas
classes = ["With Mask", "Without Mask"]


@app.route("/")
def home():
    return render_template(
        "index.html",
        result=None,
        confidence=None,
        image=None
    )


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            result=None,
            confidence=None,
            image=None
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            result=None,
            confidence=None,
            image=None
        )

    # ==========================
    # Simpan gambar
    # ==========================
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    # ==========================
    # Baca gambar
    # ==========================
    img = cv2.imread(filepath)

    if img is None:
        return render_template(
            "index.html",
            result="Gagal membaca gambar.",
            confidence=0,
            image=filepath
        )

    # ==========================
    # Deteksi wajah
    # ==========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # ==========================
    # Jika wajah ditemukan,
    # gunakan hasil crop.
    # Jika tidak, gunakan gambar asli.
    # ==========================
    if len(faces) > 0:

        (x, y, w, h) = faces[0]

        face = img[y:y+h, x:x+w]

    else:

        face = img

    # ==========================
    # Preprocessing CNN
    # ==========================
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face = cv2.resize(face, (128, 128))

    face = face.astype("float32") / 255.0

    face = np.expand_dims(face, axis=0)

    # ==========================
    # Prediksi CNN
    # ==========================
    prediction = model.predict(face, verbose=0)

    label_index = np.argmax(prediction)

    confidence = float(prediction[0][label_index] * 100)

    result = classes[label_index]

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)