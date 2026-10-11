import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_model.keras")

model = load_model()

with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("🌿 Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image to detect its disease.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing leaf image..."):
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img), axis=0)

            predictions = model.predict(img_array)[0]
            index = int(np.argmax(predictions))
            prediction = class_names[index]
            confidence = float(predictions[index]) * 100

        if prediction.lower() == "healthy":
            st.success(f"✅ Prediction: {prediction}")
        else:
            st.error(f"⚠️ Prediction: {prediction}")

        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Class Probabilities")

        for i, score in enumerate(predictions):
            st.write(f"**{class_names[i]}**")
            st.progress(float(score))