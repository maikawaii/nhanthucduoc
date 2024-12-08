import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoProcessor
import numpy as np

# Tải mô hình và processor từ Hugging Face
model_name = "Laimaimai/herbal_identification"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Thay đổi giao diện: Màu sắc nền và font chữ
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Arial', sans-serif;
        }
        .title {
            color: #2E8B57;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
        }
        .header {
            text-align: center;
            color: #3CB371;
            font-size: 20px;
        }
        .result {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .result p {
            font-size: 18px;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            color: #3CB371;
        }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề ứng dụng
st.markdown('<h1 class="title">Nhận diện dược liệu</h1>', unsafe_allow_html=True)

# Hướng dẫn người dùng
st.markdown('<p class="header">Chọn ảnh dược liệu để nhận diện cây và dự đoán các loại cây tương ứng.</p>', unsafe_allow_html=True)

# Nhận đầu vào ảnh từ người dùng
image = st.file_uploader("Tải lên ảnh", type=["jpg", "jpeg", "png"])

if image is not None:
    # Hiển thị ảnh tải lên
    st.image(image, caption="Ảnh tải lên", use_column_width=True)
    
    # Xử lý ảnh
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))  # Resize ảnh nếu cần

    # Chuyển ảnh thành định dạng phù hợp với mô hình
    inputs = processor(images=img, return_tensors="pt")

    # Dự đoán với mô hình
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

    # Lấy top 5 dự đoán
    top_k = torch.topk(probs, 5)
    labels = top_k.indices[0].numpy()
    scores = top_k.values[0].numpy()

    # Hiển thị kết quả top 5 dự đoán
    st.markdown('<div class="result">', unsafe_allow_html=True)
    st.write("Top 5 cây dược liệu dự đoán:")
    for i in range(5):
        st.write(f"{i+1}. Cây {labels[i]} - {scores[i]*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# Thêm footer (chân trang)
st.markdown('<p class="footer">Ứng dụng nhận diện dược liệu - Streamlit</p>', unsafe_allow_html=True)
