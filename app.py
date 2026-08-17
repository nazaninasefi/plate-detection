import streamlit as st
import cv2
import easyocr
import numpy as np
from PIL import Image

st.set_page_config(page_title="تشخیص پلاک خودرو", page_icon="🚗")
st.markdown("""
    <style>
    .block-container { text-align: center; }
    .stImage { display: flex; justify-content: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 سیستم تشخیص پلاک خودرو")
st.write("عکس خودرو رو آپلود کن تا پلاکش خونده بشه")

uploaded_file = st.file_uploader("عکس خودرو را انتخاب کنید", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="تصویر آپلود شده", use_column_width=True)

    with st.spinner("در حال پردازش تصویر..."):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plate = None
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(c)
                plate = img_rgb[y:y+h, x:x+w]
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)
                break

        reader = easyocr.Reader(['fa', 'en'])

        if plate is not None:
            result = reader.readtext(plate)
            st.image(plate, caption="پلاک استخراج شده", use_column_width=True)
        else:
            result = reader.readtext(img_rgb)
            st.warning("پلاک به صورت دقیق تشخیص داده نشد، کل تصویر بررسی شد")

    st.image(img_rgb, caption="تصویر با پلاک مشخص شده", use_column_width=True)

st.markdown("""
    <style>
    .main { text-align: center; }
    .plate-text { 
        font-size: 48px; 
        font-weight: bold; 
        text-align: center;
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 10px 0;
    }
    .detail-text {
        font-size: 14px;
        text-align: center;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

st.subheader("📋 نتیجه خواندن پلاک:")
if result:
    for detection in result:
        text = detection[1]
        confidence = detection[2]
        st.markdown(f"""
            <div style='text-align:center'>
                <p style='font-size:42px; font-weight:bold; margin-bottom:4px'>{text}</p>
                <p style='font-size:14px; color:gray'>دقت: {confidence:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.error("متنی تشخیص داده نشد")