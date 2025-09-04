import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# Judul aplikasi
st.set_page_config(page_title="Klasifikasi Anjing vs Kucing", page_icon="🐶🐱")
st.title("🐶🐱 Klasifikasi Anjing vs Kucing")

# === Download & Load Model ===
MODEL_REPO = "zahratalitha/anjingkucing"   # ganti dengan repo HuggingFace kamu
MODEL_FILE = "kucinganjing.keras"          # pastikan sesuai nama file di repo

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

# load model
model = tf.keras.models.load_model(model_path, compile=False)
st.success("✅ Model berhasil dimuat!")
st.write("Input shape model:", model.input_shape)

# === Fungsi preprocessing ===
def preprocess(img: Image.Image):
    # pastikan RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # resize sesuai input model
    target_size = tuple(model.input_shape[1:3])
    if img.size != target_size:
        img = img.resize(target_size)

    # normalisasi
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # tambahkan batch dimensi
    return img_array

# === Upload Gambar ===
uploaded_file = st.file_uploader("📤 Upload gambar anjing/kucing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing
    input_img = preprocess(image)

    # Prediksi
    pred = model.predict(input_img)

    # Jika output 1 neuron (sigmoid)
    if pred.shape[1] == 1:
        prob = float(pred[0][0])
        label = "🐱 Kucing" if prob < 0.5 else "🐶 Anjing"
        confidence = 1 - prob if prob < 0.5 else prob
    else:
        # Jika output 2 neuron (softmax)
        class_idx = np.argmax(pred[0])
        label = "🐱 Kucing" if class_idx == 0 else "🐶 Anjing"
        confidence = float(np.max(pred[0]))

    # Tampilkan hasil
    st.subheader(f"Prediksi: {label}")
    st.write(f"Confidence: **{confidence:.2f}**")
