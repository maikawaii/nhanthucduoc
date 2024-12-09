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

# Tải file ánh xạ mã -> tên tiếng Việt
mapping_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/main/label_vietnamese.txt"
response_mapping = requests.get(mapping_url)
label_mapping = {}

if response_mapping.status_code == 200:
    mapping_data = response_mapping.text.splitlines()
    for line in mapping_data:
        key, value = line.split("=", 1)  # Phân tách mã và tên tiếng Việt
        label_mapping[key.strip()] = value.strip()
else:
    st.error("Không thể tải file ánh xạ mã sang tên tiếng Việt.")

# Tải file label_info.txt
info_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/main/label_info.txt"
response_info = requests.get(info_url)
plant_info = {}
plant_images = {}  # Lưu URL hình ảnh cho mỗi cây

if response_info.status_code == 200:
    info_data = response_info.text.splitlines()
    current_plant = None
    current_info = []
    current_image_url = None

    for line in info_data:
        if any(line.startswith(label) for label in labels):
            if current_plant:
                plant_info[current_plant] = "\n".join(current_info)
                if current_image_url:
                    plant_images[current_plant] = current_image_url
            current_plant = line.strip()
            current_info = []
            current_image_url = None  # Reset URL hình ảnh
        elif line.startswith("Hình ảnh:"):
            current_image_url = line.split(":")[1].strip()  # Lấy URL hình ảnh
        current_info.append(line.strip())

    if current_plant:
        plant_info[current_plant] = "\n".join(current_info)
        if current_image_url:
            plant_images[current_plant] = current_image_url
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
            label_code = labels[top_5_indices[i].item()]
            label_vietnamese = label_mapping.get(label_code, label_code)  # Lấy tên tiếng Việt hoặc mã gốc
            confidence = top_5_confidences[i].item()

            # Accordion để mở thông tin chi tiết cây
            with st.expander(f"{i + 1}. {label_vietnamese} ({confidence:.2f}%)"):
                plant_details = plant_info.get(label_code, "Không có thông tin chi tiết cho cây này.")
                plant_details = plant_details.split("\n")
                for detail in plant_details:
                    st.write(detail)

                # Hiển thị hình ảnh cây (nếu có)
                if label_code in plant_images:
                    # Đảm bảo URL hình ảnh từ Google Drive có dạng https://drive.google.com/uc?id=ID_HÌNH_ẢNH
                    image_url = plant_images[label_code]
                    if "drive.google.com" in image_url:
                        # Sửa lại URL để lấy ảnh trực tiếp từ Google Drive
                        image_url = image_url.replace("https://drive.google.com/file/d/", "https://drive.google.com/uc?id=").split("/view")[0]
                    st.image(image_url, caption=f"Hình ảnh của {label_vietnamese}")

# Trang đối chiếu
elif page == "Trang đối chiếu":
    st.title("Thông tin Dược liệu")

    if labels and plant_info:
        # Chuyển labels thành danh sách tên tiếng Việt
        vietnamese_labels = [label_mapping.get(label, label) for label in labels]
        selected_plant = st.selectbox("Chọn cây để xem thông tin:", options=vietnamese_labels)

        # Tìm mã cây tương ứng với tên tiếng Việt
        selected_label_code = next((k for k, v in label_mapping.items() if v == selected_plant), None)

        # Hiển thị thông tin cây được chọn
        if selected_label_code:
            plant_details = plant_info.get(selected_label_code, "Không có thông tin cho cây này.")
            st.subheader(selected_plant)
            plant_details = plant_details.split("\n")
            for detail in plant_details:
                st.write(detail)

            # Hiển thị hình ảnh cây (nếu có)
            if selected_label_code in plant_images:
                image_url = plant_images[selected_label_code]
                if "drive.google.com" in image_url:
                    # Sửa lại URL để lấy ảnh trực tiếp từ Google Drive
                    image_url = image_url.replace("https://drive.google.com/file/d/", "https://drive.google.com/uc?id=").split("/view")[0]
                st.image(image_url, caption=f"Hình ảnh của {selected_plant}")
        else:
            st.warning("Không tìm thấy thông tin cây được chọn.")
    else:
        st.error("Dữ liệu cây hoặc thông tin cây chưa sẵn sàng.")
