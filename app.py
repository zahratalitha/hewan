import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Klasifikasi Anjing vs Kucing", page_icon="🐶🐱")
st.title("🐶🐱 Klasifikasi Anjing vs Kucing")

# === Download & Load Model ===
MODEL_REPO = "zahratalitha/anjingkucing"   # ganti dengan repo HuggingFace kamu
MODEL_FILE = "kucinganjing.keras"          # pastikan sesuai nama file di repo

with st.spinner("📥 Downloading model..."):
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

model = tf.keras.models.load_model(model_path, compile=False)
st.success("✅ Model berhasil dimuat!")
st.write("Input shape model:", model.input_shape)

# === Fungsi Preprocessing ===
def preprocess(img: Image.Image):
    # ✅ Paksa RGB (3 channel)
    img = img.convert("RGB")

    # ✅ Tentukan target size (ambil dari model, default 224x224 kalau None)
    if model.input_shape[1] is None or model.input_shape[2] is None:
        target_size = (224, 224)
    else:
        target_size = (model.input_shape[1], model.input_shape[2])

    # Resize
    img = img.resize(target_size)

    # Normalisasi
    img_array = np.asarray(img, dtype=np.float32) / 255.0

    # ✅ Pastikan bentuk (H, W, 3)
    if img_array.ndim == 2:  # grayscale edge-case
        img_array = np.stack([img_array] * 3, axis=-1)

    # Tambahkan batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === Upload Gambar ===
uploaded_file = st.file_uploader("📤 Upload gambar anjing/kucing", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Gambar yang diupload", use_column_width=True)

    # Preprocessing
    input_img = preprocess(image)

    # Prediksi
    pred = model.predict(input_img)

    # === Handling Output ===
    if pred.shape[1] == 1:  # sigmoid
        prob = float(pred[0][0])
        label = "🐱 Kucing" if prob < 0.5 else "🐶 Anjing"
        confidence = 1 - prob if prob < 0.5 else prob
    else:  # softmax
        class_idx = np.argmax(pred[0])
        label = "🐱 Kucing" if class_idx == 0 else "🐶 Anjing"
        confidence = float(np.max(pred[0]))

    # === Tampilkan hasil ===
    st.subheader(f"Prediksi: {label}")
    st.write(f"Confidence: **{confidence:.2f}**")
