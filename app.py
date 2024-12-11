import streamlit as st
import requests
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image
from io import BytesIO

# Hàm tải hình ảnh từ URL
def load_image_from_url(image_url):
    try:
        if "drive.google.com" in image_url:
            image_url = image_url.replace("https://drive.google.com/file/d/", "https://drive.google.com/uc?id=").split("/view")[0]
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            st.warning(f"Lỗi tải hình ảnh: {response.status_code} - {image_url}")
            return None
    except Exception as e:
        st.warning(f"Không thể tải hình ảnh: {e}")
        return None

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
labels_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/refs/heads/main/label.txt"
response_labels = requests.get(labels_url)
labels = response_labels.text.splitlines() if response_labels.status_code == 200 else []
if not labels:
    st.error("Không thể tải labels.txt từ GitHub.")

# Tải file ánh xạ mã -> tên tiếng Việt
mapping_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/main/label_vietnamese.txt"
response_mapping = requests.get(mapping_url)
label_mapping = {}
if response_mapping.status_code == 200:
    for line in response_mapping.text.splitlines():
        key, value = line.split("=", 1)
        label_mapping[key.strip()] = value.strip()
else:
    st.error("Không thể tải file ánh xạ mã sang tên tiếng Việt.")

# Tải file label_info.txt (chứa thông tin cây thuốc)
info_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/main/label_info.txt"
response_info = requests.get(info_url)
plant_info = {}
if response_info.status_code == 200:
    current_label = None
    for line in response_info.text.splitlines():
        if any(line.startswith(label) for label in labels):
            current_label = line.strip()
            plant_info[current_label] = {"description": ""}
        elif current_label:
            plant_info[current_label]["description"] += f" {line.strip()}"
else:
    st.error("Không thể tải label_info.txt từ GitHub.")

# Tải file imagine_info.txt (chứa URL ảnh)
image_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/refs/heads/main/image_info.txt"
response_image = requests.get(image_url)
image_info = {}
if response_image.status_code == 200:
    for line in response_image.text.splitlines():
        key, value = line.split(":", 1)
        image_info[key.strip()] = value.strip()
else:
    st.error("Không thể tải imagine_info.txt từ GitHub.")

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
        # Hiển thị ảnh đã tải lên
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        # Dự đoán
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        # Lấy top 1 kết quả
        top_1_idx = torch.argmax(logits, dim=-1).item()
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][top_1_idx].item() * 100

        if confidence < 50:  # Ngưỡng xác suất
            st.warning("Không nhận diện được cây nào khớp với ảnh này.")
        else:
            label_code = labels[top_1_idx]
            plant_name = label_mapping.get(label_code, label_code)
            plant_details = plant_info.get(label_code, {})
            plant_description = plant_details.get("description", "Không có thông tin.")
            plant_image_url = image_info.get(label_code, None)

            col1, col2 = st.columns([1, 2])
            with col1:
                if plant_image_url:
                    img = load_image_from_url(plant_image_url)
                    if img:
                        st.image(img, caption=f"Hình ảnh của {plant_name}")
            with col2:
                st.subheader(plant_name)
                st.markdown(plant_description)

# Trang đối chiếu
elif page == "Trang đối chiếu":
    st.title("Thông tin Dược liệu (Tham khảo từ sách Dược liệu)")

    if labels and plant_info:
        vietnamese_labels = [label_mapping.get(label, label) for label in labels]
        selected_plant = st.selectbox("Chọn cây để xem thông tin:", options=vietnamese_labels)

        selected_label_code = next((k for k, v in label_mapping.items() if v == selected_plant), None)

        if selected_label_code:
            plant_details = plant_info.get(selected_label_code, {})
            plant_name = label_mapping.get(selected_label_code, "Không rõ")
            plant_description = plant_details.get("description", "Không có thông tin.")
            plant_image_url = image_info.get(selected_label_code, None)

            col1, col2 = st.columns([1, 2])
            with col1:
                if plant_image_url:
                    img = load_image_from_url(plant_image_url)
                    if img:
                        st.image(img, caption=f"Hình ảnh của {plant_name}")
            with col2:
                st.subheader(plant_name)
                st.markdown(plant_description)
