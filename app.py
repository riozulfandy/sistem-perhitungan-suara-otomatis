import streamlit as st
from inference_sdk import InferenceHTTPClient
from paddleocr import PaddleOCR
from PIL import Image, ExifTags
import time
import cv2
import numpy as np

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


# ====================================FUNCTION=====================================

def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
    return image

def crop_image(img, bounding_boxes):
    # Bird view cropping
    xmin, ymin, xmax, ymax = bounding_boxes
    points = [(xmin, ymax), (xmin, ymin), (xmax, ymin), (xmax, ymax)]
    pts1 = np.float32([[points[0][0], points[0][1]], [points[1][0], points[1][1]],
                          [points[2][0], points[2][1]], [points[3][0], points[3][1]]])
    
    width = 200
    height = 400

    pts2 = np.float32([[0, height], [0, 0], [width, 0], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (width, height))

    return img_output

def get_bounding_boxes(img, CLIENT):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()
    result = CLIENT.infer(gray_image, model_id="pemilu-gemastik/1")
    end = time.time()
    api_latency = (end-start) - result['time']
    
    bounding_boxes = result['predictions']
    inference_time = result['time']

    dict_bounding_boxes = {}
    for bounding_boxes in bounding_boxes:
        x1 = bounding_boxes['x'] - bounding_boxes['width'] / 2
        x2 = bounding_boxes['x'] + bounding_boxes['width'] / 2
        y1 = bounding_boxes['y'] - bounding_boxes['height'] / 2
        y2 = bounding_boxes['y'] + bounding_boxes['height'] / 2
        class_name = bounding_boxes['class']
        box = (x1, y1, x2, y2)
        dict_bounding_boxes[class_name] = box
    return dict_bounding_boxes, inference_time, api_latency

def get_texts(ocr_model, img):
    start = time.time()
    result = ocr_model.ocr(img, cls=False,det=False)
    end = time.time()

    return result[0][0][0], result[0][0][1], end-start

def process(img, CLIENT, ocr):
    # Get the bounding boxes
    bounding_boxes, inference_time, api_latency = get_bounding_boxes(img, CLIENT)

    # Predict the texts
    result_dict = {}
    for no_urut in range(1,4):
        bounding_box = bounding_boxes[f"{no_urut}"]
        cropped_image = crop_image(img, bounding_box)
        text, confidence, ocr_time = get_texts(ocr, cropped_image)
        result_dict[no_urut] = {"text": text, "confidence": confidence, "ocr_time": ocr_time}

    result_dict['detect_time'] = inference_time
    result_dict['yolo_api_latency'] = api_latency

    return result_dict

def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image)
    # Konversi dari RGB (PIL) ke BGR (OpenCV)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

st.title("Sistem Perhitungan Suara Pemilu Otomatis")
st.markdown("Website ini merupakan sistem perhitungan suara pemilu otomatis menggunakan teknologi OCR dan Object Detection.")
st.info("Upload gambar suara pemilu untuk memulai perhitungan suara otomatis.")


example_images = {
    "Gambar 1": "example_images/TPS_0332.jpg",
    "Gambar 2": "example_images/TPS_0443.jpg",
    "Gambar 3": "example_images/TPS_0523.jpg",
    "Gambar 4": "example_images/TPS_0631.jpg",
    "Gambar 5": "example_images/TPS_0664.jpg",
    "Gambar 6": "example_images/TPS_0713.jpg",
    "Gambar 7": "example_images/TPS_0864.jpg",
    "Gambar 8": "example_images/TPS_1000.jpg",
    "Gambar 9": "example_images/TPS_1105.jpg",
    "Gambar 10": "example_images/TPS_1111.jpg",
    "Gambar 11": "example_images/TPS_0060.jpg",
    "Gambar 12": "example_images/TPS_1186.jpg",
}

# ====================================USER INTERFACE=====================================

example_image = st.selectbox("Pilih gambar contoh", list(example_images.keys()))
uploaded_image = st.file_uploader("Atau upload gambar suara pemilu", type=['jpg','jpeg','png'])
process_btn = st.button("Proses Gambar")


img = None
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    img = correct_image_orientation(img)
elif example_image:
    img = Image.open(example_images[example_image])
    img = correct_image_orientation(img)


if img is not None and process_btn:
    img = pil_to_cv2(img)
    start_time = time.time()
    result = process(img, CLIENT, ocr)
    end_time = time.time()
    total_time = end_time - start_time
    system_time = total_time - result['yolo_api_latency']
    del result['yolo_api_latency']
    st.markdown("### Hasil Perhitungan Suara")
    st.write(result)
    st.write("### Gambar Suara Pemilu")
    st.write(f"### Waktu Proses Sistem: {system_time:4f} detik")
    st.image(img, channels="BGR")