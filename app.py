import streamlit as st
import requests
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image

# Thêm CSS cho hình nền
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1516802273409-68526ee1bdd6");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# 1. URL của file labels.txt trên GitHub
url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/refs/heads/main/label.txt"

# 2. Tải file labels.txt về
response = requests.get(url)

if response.status_code == 200:
    labels = response.text.splitlines()
else:
    labels = None
    st.error("Không thể tải labels.txt từ GitHub. Hãy kiểm tra URL hoặc kết nối mạng.")

# 3. Tạo từ điển thông tin cây
# Thay thông tin này bằng thông tin chi tiết về 78 loại cây
plant_info = {
    label: f"Thông tin về {label}: Đây là cây thuốc có tác dụng..." for label in labels
}

# 4. Tải mô hình và processor từ Hugging Face
model_name = "Laimaimai/herbal_identification"
try:
    model = AutoModelForImageClassification.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
except Exception as e:
    st.error(f"Không thể tải mô hình từ Hugging Face: {e}")
    model, processor = None, None

# 5. Tạo hai nút điều hướng
page = st.sidebar.selectbox("Chọn trang", ["Trang chủ", "Trang đối chiếu"])

if page == "Trang chủ":
    st.title("Ứng dụng Nhận diện Dược Liệu")

    # Tải ảnh từ người dùng
    uploaded_file = st.file_uploader("Chọn hình ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None and labels:
        try:
            # Mở ảnh
            image = Image.open(uploaded_file)

            # Hiển thị ảnh
            st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

            # Tiền xử lý ảnh và dự đoán
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                logits = model(**inputs).logits

            # Lấy top 1 kết quả dự đoán
            top_1_index = torch.argmax(logits, dim=-1).item()
            predicted_label = labels[top_1_index]
            confidence = torch.nn.functional.softmax(logits, dim=-1)[0][top_1_index].item() * 100

            # Hiển thị kết quả dự đoán
            st.success(f"Dự đoán: {predicted_label} ({confidence:.2f}%)")

            # Lưu nhãn để sử dụng trong trang đối chiếu
            st.session_state["selected_label"] = predicted_label
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi xử lý ảnh hoặc dự đoán: {e}")
    else:
        st.info("Vui lòng tải lên một hình ảnh.")

elif page == "Trang đối chiếu":
    st.title("Trang đối chiếu thông tin cây")

    # Hiển thị danh sách các cây
    if "selected_label" in st.session_state:
        selected_label = st.session_state["selected_label"]
        st.write(f"Bạn đã chọn: **{selected_label}**")

        # Hiển thị thông tin cây
        if selected_label in plant_info:
            st.write(plant_info[selected_label])
        else:
            st.write("Không tìm thấy thông tin cây này.")
    else:
        st.info("Bạn cần dự đoán một cây trước khi truy cập trang này.")
