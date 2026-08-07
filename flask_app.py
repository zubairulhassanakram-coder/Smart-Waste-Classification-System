import os
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename

# ------------------- Config -------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_classifier_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASSES = ["Cardboard", "Glass", "Metal"]
IMG_SIZE = 128

# ------------------- Load Model -------------------
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASSES[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)

    probabilities = {
        CLASSES[i]: round(float(predictions[0][i]) * 100, 2)
        for i in range(len(CLASSES))
    }

    return predicted_class, confidence, probabilities


# ------------------- Routes -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if model is None:
        return render_template(
            "index.html",
            error="Model file not found. Please train the model first "
                  "(run 02_model_training.ipynb) and make sure "
                  "models/waste_classifier_model.h5 exists.",
        )

    if request.method == "POST":
        file = request.files.get("file")

        if file is None or file.filename == "":
            return render_template("index.html", error="No file selected.")

        if not allowed_file(file.filename):
            return render_template("index.html", error="Please upload a .jpg, .jpeg, or .png file.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        predicted_class, confidence, probabilities = predict_image(filepath)

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=round(confidence, 2),
            probabilities=probabilities,
            image_path=f"uploads/{filename}",
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
