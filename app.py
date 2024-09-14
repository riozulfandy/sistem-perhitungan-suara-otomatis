import streamlit as st
from inference_sdk import InferenceHTTPClient
from paddleocr import PaddleOCR
from backend import process, pil_to_cv2
from PIL import Image

st.set_page_config(page_title="Sistem Pembaca Perhitungan Suara Pemilu Otomatis", page_icon="📝", layout="centered")

# ====================================LOAD MODELS=====================================
@st.cache_resource
def load_detection_model():
    return InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="AJt9IJDh0ntOGvlSeklT"
    )

@st.cache_resource
def load_ocr_model():
    return PaddleOCR(rec_model_dir='./model_inference', lang='en',ocr_version='PP-OCRv3',det=False, cls=False,show_log=False)

CLIENT = load_detection_model()
ocr = load_ocr_model()

# ====================================USER INTERFACE=====================================

st.title("Sistem Perhitungan Suara Pemilu Otomatis")
st.markdown("Website ini merupakan sistem perhitungan suara pemilu otomatis menggunakan teknologi OCR dan Object Detection.")
st.info("Upload gambar suara pemilu untuk memulai perhitungan suara otomatis.")


example_images = {
    "Gambar 1": "example_images/TPS_0120.jpg",
    "Gambar 2": "example_images/TPS_0188.jpg",
    "Gambar 3": "example_images/TPS_1011.jpg",
    "Gambar 4": "example_images/TPS_0001.jpg",
    "Gambar 5": "example_images/TPS_0123.jpg",
    "Gambar 6": "example_images/TPS_0175.jpg",
    "Gambar 7": "example_images/TPS_0699.jpg",
}

example_image = st.selectbox("Pilih gambar contoh", list(example_images.keys()))
uploaded_image = st.file_uploader("Atau upload gambar suara pemilu", type=['jpg','jpeg','png'])
process_btn = st.button("Proses Gambar")


img = None
if uploaded_image is not None:
    img = Image.open(uploaded_image)
elif example_image:
    img = Image.open(example_images[example_image])


if img is not None and process_btn:
    img = pil_to_cv2(img)
    start_time = time.time()
    result = process(img, CLIENT, ocr)
    end_time = time.time()


    st.markdown("### Hasil Perhitungan Suara")
    st.write(result)
    st.write("### Gambar Suara Pemilu")
    st.write(f"### Waktu Proses Sistem: {end_time - start_time:.2f} detik")
    st.image(img, channels="BGR")


