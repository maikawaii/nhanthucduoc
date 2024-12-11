import streamlit as st
import requests
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image
from io import BytesIO

# Hàm tải hình ảnh từ URL
def load_image_from_url(image_url):
    try:
        # Kiểm tra xem URL có phải Google Drive không và sửa lại thành URL tải xuống
        if "drive.google.com" in image_url:
            image_url = image_url.replace("https://drive.google.com/file/d/", "https://drive.google.com/uc?id=").split("/view")[0]
        
        # Tải ảnh từ URL
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
        key, value = line.split("=", 1)
        label_mapping[key.strip()] = value.strip()
else:
    st.error("Không thể tải file ánh xạ mã sang tên tiếng Việt.")

# Tải file imagine_info chứa URL ảnh cho từng cây
imagine_info_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/main/imagine_info.txt"
response_imagine_info = requests.get(imagine_info_url)
imagine_info = {}

if response_imagine_info.status_code == 200:
    imagine_data = response_imagine_info.text.splitlines()
    for line in imagine_data:
        plant_code, image_url = line.split(":", 1)
        imagine_info[plant_code.strip()] = image_url.strip()
else:
    st.error("Không thể tải imagine_info.txt từ GitHub.")

# Tải file mô tả cây (hoặc lấy từ nguồn khác)
description_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/main/plant_descriptions.txt"
response_description = requests.get(description_url)
descriptions = {}

if response_description.status_code == 200:
    description_data = response_description.text.splitlines()
    for line in description_data:
        plant_code, description = line.split(":", 1)
        descriptions[plant_code.strip()] = description.strip()
else:
    st.error("Không thể tải file mô tả cây từ GitHub.")

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

        if top_5_confidences[0].item() < 50:  # Ngưỡng xác suất
            st.warning("Không nhận diện được cây nào khớp với ảnh này.")
        else:
            # Hiển thị top 5 kết quả
            st.write("**Top 5 cây dự đoán:**")
            for i in range(5):
                label_code = labels[top_5_indices[i].item()]
                
                # Lấy tên cây từ label_mapping (hoặc dùng label_code nếu không có trong label_mapping)
                plant_name_vietnamese = label_mapping.get(label_code, label_code)  # Tên cây tiếng Việt
                
                # Lấy thông tin URL ảnh từ imagine_info
                plant_image_url = imagine_info.get(label_code, None)
                
                # Lấy mô tả từ descriptions
                plant_description = descriptions.get(label_code, "Không có mô tả chi tiết.")

                with st.expander(f"{i + 1}. {plant_name_vietnamese} ({top_5_confidences[i].item():.2f}%)"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if plant_image_url:
                            img = load_image_from_url(plant_image_url)
                            if img:
                                st.image(img, caption=f"Hình ảnh của {plant_name_vietnamese}")
                    with col2:
                        st.write(f"**Tên cây**: {plant_name_vietnamese}")
                        st.write(f"**Mô tả**: {plant_description}")

# Trang đối chiếu
elif page == "Trang đối chiếu":
    st.title("Thông tin Dược liệu (Tham khảo từ sách Dược liệu)")

    if labels and imagine_info:
        vietnamese_labels = [label_mapping.get(label, label) for label in labels]
        selected_plant = st.selectbox("Chọn cây để xem thông tin:", options=vietnamese_labels)

        selected_label_code = next((k for k, v in label_mapping.items() if v == selected_plant), None)

        if selected_label_code:
            plant_name = label_mapping.get(selected_label_code, "Không rõ")
            plant_image_url = imagine_info.get(selected_label_code, None)
            plant_description = descriptions.get(selected_label_code, "Không có mô tả chi tiết.")

            col1, col2 = st.columns([1, 2])
            with col1:
                if plant_image_url:
                    img = load_image_from_url(plant_image_url)
                    if img:
                        st.image(img, caption=f"Hình ảnh của {plant_name}")
            with col2:
                st.subheader(plant_name)
                st.markdown(f"**Mô tả:** {plant_description}")
