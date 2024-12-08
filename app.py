import streamlit as st
from transformers import AutoModelForImageClassification, AutoProcessor
from PIL import Image
import torch

# Tải mô hình và processor
MODEL_NAME = "Laimaimai/herbal_identification"
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Thiết lập giao diện
st.set_page_config(page_title="Nhận diện Dược liệu", page_icon="🌿", layout="wide")

# Header
st.title("🌱 Nhận diện Dược liệu")
st.write("Tải lên ảnh của bạn và xem kết quả dự đoán top 5 cây kèm phần trăm chính xác.")

# Tải ảnh từ người dùng
uploaded_file = st.file_uploader("Tải lên một ảnh dược liệu:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn vừa tải lên.", use_column_width=True)

    # Xử lý ảnh và dự đoán
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    top_5 = torch.topk(probs, k=5)

    # Hiển thị kết quả
    st.write("### Kết quả dự đoán:")
    for i in range(5):
        st.write(f"**{processor.id2label[top_5.indices[0][i].item()]}**: {top_5.values[0][i].item() * 100:.2f}%")
