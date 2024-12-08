import streamlit as st
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image

# Tải mô hình và processor từ Hugging Face
model_name = "Laimaimai/herbal_identification"  # Tên mô hình của bạn trên Hugging Face
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Lấy labels từ mô hình
labels = model.config.id2label  # id2label là từ điển từ id -> tên label (tên cây)

# Tiêu đề với màu sắc và kích thước
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Ứng dụng nhận dạng dược liệu</h1>", unsafe_allow_html=True)

# Tiêu đề phụ
st.markdown("<h3 style='text-align: center; color: #FF6347;'>Chọn ảnh để nhận diện cây dược liệu</h3>", unsafe_allow_html=True)

# Thêm background đẹp
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F8FF;
    }
    </style>
    """, unsafe_allow_html=True
)

# Tải ảnh từ người dùng
uploaded_file = st.file_uploader("Chọn ảnh dược liệu", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Mở ảnh đã tải lên
    image = Image.open(uploaded_file)

    # Xử lý ảnh và thực hiện dự đoán
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Lấy top 5 kết quả dự đoán
    logits = outputs.logits
    top_5 = torch.topk(logits, 5)

    # Hiển thị tên cây và độ tự tin (phần trăm)
    st.markdown("<h2 style='color: #008080;'>Top 5 cây dự đoán:</h2>", unsafe_allow_html=True)
    for i in range(5):
        label = labels[top_5.indices[0][i].item()]  # Lấy tên cây từ labels
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][top_5.indices[0][i].item()].item() * 100
        st.write(f"<b>{label}</b>: {confidence:.2f}%", unsafe_allow_html=True)
