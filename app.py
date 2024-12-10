import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Hàm tải ảnh từ URL
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

# Thông tin về Tục đoạn
st.title("Tục đoạn (Dipsacus japonicus Miq.)")
st.write("### Tên khoa học: Dipsacus japonicus Miq.")
st.write("#### Đặc điểm nhận thức chính:")
st.write("Rễ hình trụ, hơi cong queo. Mặt ngoài màu nâu nhạt, có nhiều nếp nhăn và rãnh dọc. Dễ bẻ gãy, mặt bẻ lờm chởm. Mặt cắt ngang có lớp bần mỏng, tầng sinh libe- gỗ màu nâu, bó libe-gỗ màu nâu nhạt, sắp xếp thành tia tỏa ra. Mùi nhẹ, vị đắng hơi ngọt và se.")
st.write("#### Thành phần hóa học: Saponin.")
st.write("#### Công dụng: Chữa gãy xương, đứt gân.")

# Tải và hiển thị ảnh từ Google Drive
image_url = "https://drive.google.com/uc?id=1U2dNjrMr9Tats0t1m0Nqrnk-8iihYOZp"
image = load_image_from_url(image_url)
if image:
    st.image(image, caption="Hình ảnh Tục đoạn", use_column_width=True)
