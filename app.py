import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ------------------- Page Config -------------------
st.set_page_config(page_title="Smart Waste Classification", page_icon="♻️", layout="centered")

# ------------------- Constants -------------------
CLASSES = ["Cardboard", "Glass", "Metal"]
IMG_SIZE = 128

# Build an absolute path to the model so it works no matter where the
# app is launched from (local machine, GitHub, or Streamlit Cloud).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_classifier_model.h5")

# ------------------- Load Model (cached) -------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ------------------- UI -------------------
st.title("♻️ Smart Waste Classification System")
st.write("Upload an image of waste material and the CNN model will classify it as **Cardboard**, **Glass**, or **Metal**.")

if model is None:
    st.error(
        "Model file not found at `models/waste_classifier_model.h5`.\n\n"
        "Run `02_model_training.ipynb` first to train and save the model, "
        "then make sure `models/waste_classifier_model.h5` is committed to your GitHub repo "
        "before deploying to Streamlit Cloud."
    )
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Classifying..."):
        predictions = model.predict(img_array)
        predicted_class = CLASSES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    st.success(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Class Probabilities")
    for cls, prob in zip(CLASSES, predictions[0]):
        st.write(f"{cls}: {prob * 100:.2f}%")
        st.progress(float(prob))
else:
    st.info("Please upload an image to get a prediction.")

st.markdown("---")
st.caption("Smart Waste Classification System | CNN | TrashNet Dataset")
