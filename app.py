import streamlit as st
import requests
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image

# Thêm CSS để hiển thị hình nền
forest_image_url = "https://images.pexels.com/photos/2318554/pexels-photo-2318554.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"  # Link hình nền
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
        if any(line.startswith(label) for label in labels):  # Kiểm tra nếu là nhãn cây
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

# Lưu trạng thái của trang và cây được chọn
if "selected_plant" not in st.session_state:
    st.session_state["selected_plant"] = None

# Giao diện chính
st.sidebar.title("Vui lòng chọn trang:")
page = st.sidebar.radio("Chọn trang:", ["Trang chủ", "Trang đối chiếu"])

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
            if st.button(f"{i + 1}. {label} ({confidence:.2f}%)"):
                st.session_state["selected_plant"] = label
                st.experimental_rerun()

# Trang đối chiếu
elif page == "Trang đối chiếu":
    st.title("Thông tin Dược liệu")

    selected_plant = st.session_state["selected_plant"]

    if selected_plant:
        st.subheader(selected_plant)
        plant_details = plant_info.get(selected_plant, "Không có thông tin chi tiết cho cây này.")
        for line in plant_details.split("\n"):
            st.write(line)

        # Nút quay lại trang chủ
        if st.button("Quay lại Trang chủ"):
            st.session_state["selected_plant"] = None
            st.experimental_rerun()
    else:
        st.warning("Không có cây nào được chọn. Vui lòng trở lại Trang chủ để dự đoán.")

