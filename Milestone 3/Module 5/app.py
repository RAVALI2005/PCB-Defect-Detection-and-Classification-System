import streamlit as st
import io
import pandas as pd
import cv2
from PIL import Image
from backend import load_model, detect_defects

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="PCB Defect Detection", layout="wide")

# ==============================
# CSS STYLING
# ==============================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.main {
    background: linear-gradient(135deg, #0e1117, #111827);
    color: white;
}
h1 {
    text-align: center;
    color: #00FFFF;
    font-size: 42px;
    font-weight: 700;
    text-shadow: 0px 0px 20px rgba(0,255,255,0.6);
}
h2, h3 {
    color: #00FFFF;
}
label {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 600;
}
.stFileUploader {
    border: 2px dashed #00FFFF;
    padding: 20px;
    border-radius: 12px;
    background-color: #1c1f26;
}
.stFileUploader:hover {
    box-shadow: 0 0 15px rgba(0,255,255,0.4);
}
.stButton>button {
    background: linear-gradient(90deg, #00FFFF, #00bcd4);
    color: black;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
div[data-testid="stFileUploader"] button {
    background: black !important;
    color: white !important;
    border-radius: 10px;
    border: 1px solid #00FFFF;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    background: #1c1f26;
    color: white;
    font-size: 18px;
    border-left: 5px solid #00FFFF;
    margin-top: 10px;
}
img {
    border-radius: 10px;
    border: 2px solid #00FFFF;
    margin-top: 10px;
}
.custom-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #1c1f26;
    color: white !important;
}
.custom-table th {
    background: #00FFFF;
    color: black !important;
    padding: 12px;
}
.custom-table td {
    padding: 10px;
    border-bottom: 1px solid #333;
    color: white !important;
}
.custom-table tr:hover {
    background-color: #2a2e39;
}
/* ===== FILE NAME TEXT COLOR ===== */
/* Drag & drop text */
div[data-testid="stFileUploader"] section {
    color: white !important;
}
/* Uploaded file name specifically */
[data-testid="stFileUploaderFileName"] {
    color: white !important;
    font-weight: 600;
}
/* Small helper text (file size & formats) */
div[data-testid="stFileUploader"] small {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================
st.markdown("<h1>PCB DEFECT DETECTION SYSTEM</h1>", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
model_path = r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\train_efficientnet_results\pcb_defect_model.pth"

@st.cache_resource
def get_model():
    return load_model(model_path)

model = get_model()

# ==============================
# UPLOAD SECTION
# ==============================
st.markdown("<h3>📂 Upload Images</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    template_file = st.file_uploader("Template PCB Image", type=["jpg","png"])

with col2:
    test_file = st.file_uploader("Test PCB Image", type=["jpg","png"])

# ==============================
# PROCESS
# ==============================
if template_file and test_file:

    template = Image.open(template_file).convert("RGB")
    test = Image.open(test_file).convert("RGB")

    # Detection with spinner
    with st.spinner("🔍 Detecting defects... please wait"):
        output_img, results, defect_count, defect_name = detect_defects(
            template, test, model, test_file.name
        )

    # ==============================
    # INPUT IMAGES
    # ==============================
    st.markdown("<h3>🖼️ Input Images</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.image(template, caption="Template Image", use_container_width=True)
    col2.image(test, caption="Test Image", use_container_width=True)

    # ==============================
    # OUTPUT IMAGE
    # ==============================
    st.markdown("<h3>🧪 Detected Output</h3>", unsafe_allow_html=True)
    st.image(output_img, use_container_width=True)


    # ==============================
    # RESULTS
    # ==============================
    st.markdown("<h3>📊 Detection Results</h3>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-box">
    🔴 <b>Total Defects Found:</b> {defect_count} <br><br>
    🧠 <b>Defect Type:</b> {defect_name}
    </div>
    """, unsafe_allow_html=True)
    # ==============================
    # TABLE + CSV DOWNLOAD
    # ==============================
    st.markdown("<h3>📋 Prediction Log</h3>", unsafe_allow_html=True)

    if len(results) > 0:
        df = pd.DataFrame(results)
        df.columns = ["Defect Type", "Confidence (%)", "Area (X,Y)"]

        html_table = df.to_html(index=False, classes="custom-table", escape=False)
        st.markdown(html_table, unsafe_allow_html=True)


    
