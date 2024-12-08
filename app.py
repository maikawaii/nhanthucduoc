import streamlit as st
import requests
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image
import io

# 1. URL của file labels.txt trên GitHub
url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/refs/heads/main/label.txt"

# 2. Tải file labels.txt về
response = requests.get(url)

# Đảm bảo tải thành công
if response.status_code == 200:
    labels = response.text.splitlines()  # Đọc các tên cây từ file labels.txt
else:
    labels = []
    st.error("Không thể tải labels.txt từ GitHub.")

# 3. Tải mô hình và processor từ Hugging Face
model_name = "Laimaimai/herbal_identification"  # Thay bằng tên mô hình của bạn
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# 4. Giao diện người dùng Streamlit
st.title("Ứng dụng Nhận diện Dược Liệu")

# Tải ảnh từ người dùng
uploaded_file = st.file_uploader("Chọn hình ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mở ảnh
    image = Image.open(uploaded_file)

    # Hiển thị ảnh
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    # 5. Tiền xử lý ảnh và dự đoán
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # 6. Lấy top 5 kết quả dự đoán
    top_5 = torch.topk(logits, 5)

    # Hiển thị kết quả dự đoán
    st.write("Top 5 cây dự đoán kèm phần trăm:")
    for i in range(5):
        label = labels[top_5.indices[0][i].item()]  # Tên cây từ labels.txt
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][top_5.indices[0][i].item()].item() * 100
        st.write(f"{label}: {confidence:.2f}%")
