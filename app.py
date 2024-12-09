import streamlit as st
import requests
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image

# Thêm CSS để hiển thị hình nền
forest_image_url = "https://i.pinimg.com/736x/ee/02/7b/ee027bb6c2329be99954e4784e79899e.jpg"  # Link tới hình ảnh cánh rừng của bạn
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
if response.status_code == 200:
    labels = response.text.splitlines()  # Đọc tên cây từ file labels.txt
else:
    labels = []
    st.error("Không thể tải labels.txt từ GitHub.")

# Tải file label_info.txt
info_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/main/label_info.txt"
response_info = requests.get(info_url)
if response_info.status_code == 200:
    info_data = response_info.text.splitlines()
    plant_info = {}
    current_plant = None
    current_info = []
    
    for line in info_data:
        if any(line.startswith(label) for label in labels):  # Kiểm tra xem có phải là tên cây không
            if current_plant:  # Lưu thông tin cây trước đó vào từ điển
                plant_info[current_plant] = "\n".join(current_info)
            current_plant = line.strip()  # Lưu tên cây hiện tại
            current_info = []  # Khởi tạo lại thông tin cây
        current_info.append(line.strip())  # Thêm thông tin cây vào danh sách
    
    # Lưu cây cuối cùng
    if current_plant:
        plant_info[current_plant] = "\n".join(current_info)
    st.success("Tải thông tin cây thành công.")
else:
    plant_info = {}
    st.error("Không thể tải label_info.txt từ GitHub.")

# Tải mô hình và processor từ Hugging Face
model_name = "Laimaimai/herbal_identification"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Giao diện chính
st.sidebar.titliệu"Vui lòng dự đoán cây sau đó chuyển sang trang đối chiếu để xem thông tin")
page = st.sidebar.radio("Chọn trang:", ["Trang chủ", "Trang đối chiếu"])

# Trang chủ
if page == "Trang chủ":
    st.title("Nhận diện dược liệu")
    uploaded_file = st.file_uploader("Nhập ảnh vào đây bạn nhé, sau khi dự đoán vui lòng sang trang đối chiếu để confirm kết quả và xem thêm thông tin", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Hiển thị ảnh
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        # Dự đoán
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        # Lấy top 5 kết quả dự đoán
        top_5 = torch.topk(logits, 5)
        top_5_indices = top_5.indices[0]
        top_5_confidences = torch.nn.functional.softmax(logits, dim=-1)[0][top_5_indices] * 100

        # Hiển thị top 5 dự đoán
        st.write("**Top 5 cây dự đoán:**")
        for i in range(5):
            label = labels[top_5_indices[i].item()]
            confidence = top_5_confidences[i].item()
            st.write(f"{i + 1}. {label} ({confidence:.2f}%)")

else:
    # Trang đối chiếu - Người dùng chọn cây để đọc thông tin
    st.title("Thông tin Dược liệu")

    if labels and plant_info:
        # Hiển thị danh sách 78 cây
        selected_plant = st.selectbox("Chọn cây để xem thông tin:", options=labels)

        # Tìm thông tin cây được chọn
        plant_details = plant_info.get(selected_plant)
        
        if plant_details:
            st.subheader(selected_plant)
            plant_details = plant_details.split("\n")

            # Hiển thị thông tin cây
            for detail in plant_details:
                st.write(detail)  # Hiển thị từng dòng thông tin cây
        else:
            st.warning("Không có thông tin cho cây này.")
    else:
        st.error("Dữ liệu cây hoặc thông tin cây chưa sẵn sàng.")
