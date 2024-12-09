import streamlit as st
import requests
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image

# Thêm CSS để hiển thị hình nền
forest_image_url = "https://images.pexels.com/photos/2318554/pexels-photo-2318554.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{forest_image_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Tải file labels.txt
url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/refs/heads/main/label.txt"
response = requests.get(url)
labels = response.text.splitlines() if response.status_code == 200 else []
if not labels:
    st.error("Không thể tải labels.txt từ GitHub.")

# Tải file label_info.txt
info_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/main/label_info.txt"
response_info = requests.get(info_url)
plant_info = {}

if response_info.status_code == 200:
    info_data = response_info.text.splitlines()
    current_plant = None
    current_info = []

    for line in info_data:
        if any(line.startswith(label) for label in labels):
            if current_plant:
                plant_info[current_plant] = "\n".join(current_info)
            current_plant = line.strip()
            current_info = []
        current_info.append(line.strip())

    if current_plant:
        plant_info[current_plant] = "\n".join(current_info)
else:
    st.error("Không thể tải label_info.txt từ GitHub.")

# Tải mô hình và processor từ Hugging Face
model_name = "Laimaimai/herbal_identification"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Giao diện chính
st.sidebar.title("Vui lòng chọn trang:")
page = st.sidebar.radio("Điều hướng:", ["Trang chủ", "Trang đối chiếu"])

# Trang chủ
if page == "Trang chủ":
    st.title("Nhận diện Dược liệu")
    uploaded_file = st.file_uploader("Nhập ảnh của bạn:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Hiển thị ảnh
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        # Dự đoán
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        # Lấy top 5 kết quả
        top_5 = torch.topk(logits, 5)
        top_5_indices = top_5.indices[0]
        top_5_confidences = torch.nn.functional.softmax(logits, dim=-1)[0][top_5_indices] * 100

        # Hiển thị top 5 kết quả
        st.write("**Top 5 cây dự đoán:**")
        for i in range(5):
            label = labels[top_5_indices[i].item()]
            confidence = top_5_confidences[i].item()

            # Accordion để mở thông tin chi tiết cây
            with st.expander(f"{i + 1}. {label} ({confidence:.2f}%)"):
                plant_details = plant_info.get(label, "Không có thông tin chi tiết cho cây này.")
                plant_details = plant_details.split("\n")
                for detail in plant_details:
                    st.write(detail)

# Trang đối chiếu
elif page == "Trang đối chiếu":
    st.title("Thông tin Dược liệu")

    if labels and plant_info:
        selected_plant = st.selectbox("Chọn cây để xem thông tin:", options=labels)

        # Hiển thị thông tin cây được chọn
        plant_details = plant_info.get(selected_plant, "Không có thông tin cho cây này.")
        if plant_details:
            st.subheader(selected_plant)
            plant_details = plant_details.split("\n")
            for detail in plant_details:
                st.write(detail)
    else:
        st.error("Dữ liệu cây hoặc thông tin cây chưa sẵn sàng.")
