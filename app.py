import re
import streamlit as st
import requests
from transformers import AutoModelForImageClassification, AutoProcessor
import torch
from PIL import Image
from io import BytesIO
import datetime

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
    /* CSS cho hiệu ứng tuyết rơi */
    .snowflake {{
        position: absolute;
        top: -10px;
        z-index: 9999;
        color: white;
        font-size: 2em;
        user-select: none;
        pointer-events: none;
        animation: snowfall linear infinite;
    }}

    @keyframes snowfall {{
        to {{
            transform: translateY(100vh);
            opacity: 0;
        }}
    }}

    /* Điều chỉnh hiệu ứng cho tuyết rơi */
    .snowflake:nth-child(1) {{
        left: 5%;
        animation-duration: 7s;
        animation-delay: 0s;
        font-size: 1.5em;
    }}

    .snowflake:nth-child(2) {{
        left: 15%;
        animation-duration: 8s;
        animation-delay: 1s;
        font-size: 1.8em;
    }}

    .snowflake:nth-child(3) {{
        left: 25%;
        animation-duration: 6s;
        animation-delay: 2s;
        font-size: 2em;
    }}

    .snowflake:nth-child(4) {{
        left: 35%;
        animation-duration: 10s;
        animation-delay: 0.5s;
        font-size: 1.3em;
    }}

    .snowflake:nth-child(5) {{
        left: 45%;
        animation-duration: 9s;
        animation-delay: 3s;
        font-size: 1.6em;
    }}

    .snowflake:nth-child(6) {{
        left: 55%;
        animation-duration: 12s;
        animation-delay: 0s;
        font-size: 1.4em;
    }}

    .snowflake:nth-child(7) {{
        left: 65%;
        animation-duration: 7s;
        animation-delay: 4s;
        font-size: 1.7em;
    }}

    .snowflake:nth-child(8) {{
        left: 75%;
        animation-duration: 8s;
        animation-delay: 1s;
        font-size: 1.8em;
    }}

    .snowflake:nth-child(9) {{
        left: 85%;
        animation-duration: 10s;
        animation-delay: 2s;
        font-size: 2em;
    }}

    .snowflake:nth-child(10) {{
        left: 95%;
        animation-duration: 11s;
        animation-delay: 0.5s;
        font-size: 1.5em;
    }}

    /* Thêm nhiều bông tuyết nữa để tạo hiệu ứng dày hơn */
    .snowfall {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        pointer-events: none;
        z-index: 1;  /* Đảm bảo tuyết nằm dưới phần tử ngày tháng */
    }}

    /* Cuốn lịch */
    .calendar {{
        position: fixed;
        top: 45px;
        left: 30px;
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-family: "Arial", sans-serif;
        font-size: 18px;
        text-align: center;
    }}
    <!-- Cuốn lịch ngày tháng -->
    <div class="calendar">
        {datetime.datetime.now().strftime("%A, %B %d, %Y")}
    </div>

    <!-- Thanh điều hướng và hình con người tuyết -->
    <div class="navbar">
        <button>Trang Chủ</button>
        <button>Trang Đối Chiếu</button>
        <img src="https://media.giphy.com/media/7nU9H2tG55ByI/giphy.gif" class="snowman" />
    </div>
    </style>
    <div class="snowfall">
        <span class="snowflake">❄</span>
        <span class="snowflake">❄</span>
        <span class="snowflake">❄</span>
        <span class="snowflake">❄</span>
        <span class="snowflake">❄</span>
        <span class="snowflake">❄</span>
        <span class="snowflake">❄</span>
        <span class="snowflake">❄</span>
        <span class="snowflake">❄</span>
        <span class="snowflake">❄</span>
    </div>

    <!-- Cuốn lịch ngày tháng -->
    <div class="calendar">
        {datetime.datetime.now().strftime("%A, %B %d, %Y")}
    </div>
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

# Tải file label_info.txt
info_url = "https://raw.githubusercontent.com/maikawaii/nhanthucduoc/main/label_info.txt"
response_info = requests.get(info_url)
plant_info = {}

if response_info.status_code == 200:
    info_data = response_info.text.splitlines()
    current_plant = None
    for line in info_data:
        if any(line.startswith(label) for label in labels):  # Dòng bắt đầu bằng mã cây
            current_plant = line.strip()
            plant_info[current_plant] = {"name": "", "description": "", "image": ""}
        elif current_plant:
            if line.startswith("Tên:"):
                plant_info[current_plant]["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("Mô tả:"):
                plant_info[current_plant]["description"] += "\n\n**Mô tả:** " + line.split(":", 1)[1].strip()
            elif line.startswith("Đặc điểm nhận thức chính:"):
                plant_info[current_plant]["description"] += "\n\n**Đặc điểm nhận thức chính:** " + line.split(":", 1)[1].strip()
            elif line.startswith("Thành phần hóa học:"):
                plant_info[current_plant]["description"] += "\n\n**Thành phần hóa học:** " + line.split(":", 1)[1].strip()
            elif line.startswith("Công dụng:"):
                plant_info[current_plant]["description"] += "\n\n**Công dụng:** " + line.split(":", 1)[1].strip()
            else:
                plant_info[current_plant]["description"] += " " + line.strip()
else:
    st.error("Không thể tải label_info.txt từ GitHub.")

# Định nghĩa trực tiếp URL ảnh cho từng cây (ví dụ)
plant_image_urls = {
  "10_Tuc_doan": "https://drive.google.com/uc?id=1vS89VAmVIksRNg3d6scVPPfuv83E63Eo",
  "11_Thien_mon": "https://drive.google.com/uc?id=1rn0k0--7lmd7Fq9qSW-I02OVkesJa2ES",
  "12_Sai_ho": "https://drive.google.com/uc?id=1EjWr8xATQUEJ44_tBo-oE12XxbQRzseM",
  "13_Vien_chi": "https://drive.google.com/uc?id=1Jgw-44fZiOet5vaErrmvANsNZqbL0rQ2",
  "14_Su_quan_tu": "https://drive.google.com/uc?id=14iVPBAE2PH9sQmClWJHdAfTW4n_fFuOZ",
  "15_Bach_mao_can": "https://drive.google.com/uc?id=1Pvn0PfdntkizCq_aI4FQbckqbWoryzGb",
  "16_Cau_ky_tu": "https://drive.google.com/uc?id=1mb-FrdeT7EW6ApRWykRB1Nhv-FhIwB8q",
  "17_Do_trong": "https://drive.google.com/uc?id=1D0gdVmZpJ1zv5q8eC-5NIUEMRhLsAQyl",
  "18_Dang_sam": "https://drive.google.com/uc?id=1UKNA0kYh2HEh-goZ-v8PciVZzcBgK8qj",
  "19_Cau_tich": " https://drive.google.com/uc?id=1gE7KmbG6odrEF422cgHi7GvwKL7EdwBt",
  "1_Boi_mau": "https://drive.google.com/uc?id=16hPnjp_uL9bC9-sEKr2KLgh-Fs2ziYES",
  "20_Tho_ty_tu": "https://drive.google.com/uc?id=1gps1i471F2ywjcEVpvSKakYF6di7KfX2",
  "21_Hoang_ky": "https://drive.google.com/uc?id=1EazLDSpBkC2B1N9s3CTwDGgR9xa9U0_G",
  "22_Coi_xay": "https://drive.google.com/uc?id=1_kJ7hqLHQ7snmCCjvasmhbllF8Hwutvx",
  "23_Huyen_sam": "https://drive.google.com/uc?id=1G6JrSLQ_a-iLU8RDIb6VGDpqOsY9zG6Q",
  "24_Tang_chi": "https://drive.google.com/uc?id=1WHq4IsRjx9ami07eCeBKKG2zEkoiNJRR",
  "25_Diep_ha_chau": "https://drive.google.com/uc?id=1LlvpJjlDgNILRFidIKkW6NduYTdyl5yU",
  "26_Kim_anh": "https://drive.google.com/uc?id=1oP2g6gLiLXUFp0LJKQaXKLAlXg3QNY2l",
  "27_Cat_can": "https://drive.google.com/uc?id=1KC09cgjOastnPe22kiy_4Qys74Y3uXVd",
  "28_Co_ngot": "https://drive.google.com/uc?id=1yHZBjpn8ax9q7rn_uXhh5cOdO5cNacgk",
  "29_Cuc_hoa": "https://drive.google.com/uc?id=1q4Le55RJ5e35mVmAnOA6KH7g9XBVwGhI",
  "2_Hoe_hoa": "https://drive.google.com/uc?id=1l6s1I26GGKZyLX7Cavj4vECChQiWP0oN",
  "30_To_moc": "https://drive.google.com/uc?id=1_9M-Jhk6obKO7LPry1tfu4WDm5cZApDm",
  "31_Kim_tien_thao": "https://drive.google.com/uc?id=1hAqXdeTknmagxh-XP5QbV0TlDMH9zZHx",
  "32_Dan_sam": "https://drive.google.com/uc?id=1lFkcq14mnq63ogA0YEGz7HNPm0hgA__E",
  "33_Chi_tu": "https://drive.google.com/uc?id=1jOk4-qa7aWy1yrtGwur5plj9tybilCkd",
  "34_Ngai_Cuu": "https://drive.google.com/uc?id=1pnUnVcBRFM_Af1m58yxVIastXOCApaiu",
  "35_Sinh_dia": "https://drive.google.com/uc?id=1QbgWWtnlCRRj5bzqCbPPpRaIesFFrKAe",
  "36_Nguu_tat": "https://drive.google.com/uc?id=1hqwaVxUMMTB4BBzWW4JtUhpQsvnXRxsC",
  "37_Bach_truat": "https://drive.google.com/uc?id=1M9d2VPmZFCDpk1SpMN5GlLf6rcJjet0d",
  "38_Nhan_tran": "https://drive.google.com/uc?id=1-KPCnWKwyP34dizy7aORzohtD1tuUh2X",
  "39_Duong_quy": "https://drive.google.com/uc?id=1Ch6Yc7toJJBGTcexHGHDeEo-q943OvtQ",
  "3_Linh_chi": "https://drive.google.com/uc?id=1bBFTyl_2VD03d0ttQkMJ1IjJ-MatVXzn",
  "40_Nho_noi": "https://drive.google.com/uc?id=1OvzvIXNp-szc63HoYBBpEHdLjR5zU8Sz",
  "41_Dao_nhan": "https://drive.google.com/uc?id=1DDoEDlX1md4iABihUn7GVkPEalwiNpWn",
  "42_Cat_canh": "https://drive.google.com/uc?id=1b1XDkV1HplJb-qfbUFXijiq3R98yLbHx",
  "43_Ha_kho_thao": "https://drive.google.com/uc?id=1aiKqrpd1hu57Ux8beQoj5R7N2mzJt9DB",
  "44_Xa_tien_tu": "https://drive.google.com/uc?id=1g-bhx3iiSLbg46mPkIkWJGjUXi7Sbjyb",
  "45_Che_day": "https://drive.google.com/uc?id=1bhnhlLrbT54Rp-bXhcSkbkpxAe747qMG",
  "46_Xa_can": "https://drive.google.com/uc?id=1dZJcTONZu1Y6DdUl0UpF-8IIeG0rarO1",
  "47_Tang_diep": "https://drive.google.com/uc?id=11xpYfFTIZlUay1z0HMv9RqZ_ILuSHBw9",
  "48_Ngu_boi_tu": "https://drive.google.com/uc?id=1nboVSD3Tperpv_fzE3xci6Kkd58p13vs",
  "49_Ngu_gia_bi": "https://drive.google.com/uc?id=1kndq5e0ZRtOzVxazsdAmhh6uS2AoBvLU",
  "4_Thong_thao": "https://drive.google.com/uc?id=1nD7OagbOKtAhM4gBT0QXDGh6lX_w_-6Y",
  "50_Rau_ngo": "https://drive.google.com/uc?id=1XnjnfHOkmxT4tVGhv3c33Ca0XFQDb3iX",
  "51_Nguu_bang_tu": "https://drive.google.com/uc?id=1Qtpys8ElQAcxv5NbjjCT64FYnVGBnKcQ",
  "52_Cam_thao_dat": "https://drive.google.com/uc?id=1hX70R5fT2f32ccMQwjuLa3AfT7sQfYZl",
  "53_Dai_hoang": "https://drive.google.com/uc?id=1XTTGOqPJzrJPYzfCcZM54rCMbdvd2181",
  "54_Hoai_son": "https://drive.google.com/uc?id=1cny9s77OY2VWk3PwdN3HsAUdbFRJyPyu",
  "55_Dam_duong_hoac": "https://drive.google.com/uc?id=1Es6TbjRkjpWkP6_8D_uei8L9qhy0Wikx",
  "56_Moc_qua": "https://drive.google.com/uc?id=1wkVZPLTnodhJsvEh6R275Svq_lD-nwjP",
  "57_Bo_cong_anh": "https://drive.google.com/uc?id=14EZOsFs6Tf81pxhESObUOhEyCehBkHQf",
  "58_Tho_phuc_linh": "https://drive.google.com/uc?id=1iKI4gRQQVDeSznK9NV94rzrcxW8E1kfy",
  "59_Mach_mon": "https://drive.google.com/uc?id=16Q3X-2-4nz549e8VX0Dqlf47JK86wDKP",
  "5_Trach_ta": "https://drive.google.com/uc?id=1L0Y3HYCaq9HY9K0y8n5i3usl5RmZcOrS",
  "60_Ke_dau_ngua": "https://drive.google.com/uc?id=1j0T21mpofhx2TDdCTFLKtwDE6CY1WxaG",
  "61_Tang_bach_bi": "https://drive.google.com/uc?id=1dpPcwldMqAfRAfoB9dEcWZWLkH_i0WpV",
  "62_Cam_thao_bac": "https://drive.google.com/uc?id=1hwoYPW23v8yBpcxWLWqlnxQEa99F8xH9",
  "63_O_tac_cot": "https://drive.google.com/uc?id=11rnaOcwR33UvkIqyEqyzxBVZDUKg6oiz",
  "64_Thao_quyet_minh": "https://drive.google.com/uc?id=1B0bCR5eY6QNh5Bf4Dp1395vWbi2T2Rm-",
  "65_Dai_tao": "https://drive.google.com/uc?id=1KJpuOkkV3QqnaCc30QDHCEPPE5e87j0e",
  "67_Tao_nhan": "https://drive.google.com/uc?id=1Q5yqnDxcEQcMzLqULWQbiQ4Crm9KimDO",
  "68_Ban_ha": "https://drive.google.com/uc?id=1Fe47DynUQmqg3cJuLEn3F81T3dvgfmiy",
  "69_Ca_gai_leo": "https://drive.google.com/uc?id=1JdxPnIf2UYfXuE2O9KHqANCWmNQTh1x_",
  "6_Y_di": "https://drive.google.com/uc?id=1BI6jXwuSvu7MuDOl-TrFMQMuuY_Pf2wQ",
  "70_Kho_qua": "https://drive.google.com/uc?id=11xJY0VsjX-1AR0ja84kjnbxCrzG88hFQ",
  "71_Xuyen_tam_lien": "https://drive.google.com/uc?id=15l3TPxs7ga46kBRnHv2xdYfcJBaSZkPS",
  "72_Nhan_sam": "https://drive.google.com/uc?id=16AMnBOGtfPam2e4QSoo3-EO6ndlNeagW",
  "73_Bach_gioi_tu": "https://drive.google.com/uc?id=14pR79saa4RAl7YAGx5IqhCsDtt_dd8_y",
  "74_Tam_that": "https://drive.google.com/uc?id=151ZLo9A7FuT_S354YeclamDsmECwUhBS",
  "75_Bach_chi": "https://drive.google.com/uc?id=1ErWiLoVgm8IGuJjh9r3GuQkvIcttFXpj",
  "76_Sa_sam": "https://drive.google.com/uc?id=1hHXIZvX3V9G5TFzrz3YIJ6ep1Ogibwnl",
  "77_Bach_thuoc": "https://drive.google.com/uc?id=1yk_8TXMjo2iNgZVIQi6nr5MjmhQ_9XXK",
  "78_Cam_thao_day": "https://drive.google.com/uc?id=1QO_MGV6nO1lRLRDnuCC--yToIyluvSnA",
  "7_Can_khuong": "https://drive.google.com/uc?id=1pEuadVE_U7l0hpJlt4t9UzIiSCFz55Lb",
  "8_Ty_Giai": "https://drive.google.com/uc?id=1eMMdvroU0qpK-4NzK8lx1RClDGelo1if",
  "9_Cot_toai_bo": "https://drive.google.com/uc?id=1-ig3AkD06Jz6o6E4ymVWqI9RFmYfvv83"
}

# Tải mô hình và processor từ Hugging Face
model_name = "Laimaimai/herbal_identification"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)


# Hàm thay thế phần trong ngoặc bằng in nghiêng và xóa dấu ngoặc
def italicize_latin_in_description(plant_description):
    # Sử dụng biểu thức chính quy để tìm phần trong ngoặc và thay thế bằng in nghiêng (Markdown) và xóa dấu ngoặc
    return re.sub(r"\(([^)]+)\)", r"*\1*", plant_description)

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
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
        
        # Chọn thiết bị (GPU nếu có, nếu không thì CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(torch.device("cpu"))
        model = model.to(device)  # Di chuyển mô hình vào thiết bị
        # Dự đoán
        inputs = processor(images=image, return_tensors="pt").to(device)  
        with torch.no_grad():
            logits = model(**inputs).logits

        # Lấy top 5 kết quả
        top_5 = torch.topk(logits, 5)
        top_5_indices = top_5.indices[0]
        top_5_confidences = torch.nn.functional.softmax(logits, dim=-1)[0][top_5_indices] * 100

        if top_5_confidences[0].item() < 0:  # Ngưỡng xác suất
            st.warning("Không nhận diện được cây nào khớp với ảnh này.")
        else:
            # Hiển thị top 5 kết quả
            st.write("**Top 5 cây dự đoán:**")
            for i in range(5):
                label_code = labels[top_5_indices[i].item()]
                
                # Lấy tên cây từ label_mapping (hoặc dùng label_code nếu không có trong label_mapping)
                plant_name_vietnamese = label_mapping.get(label_code, label_code)  # Tên cây tiếng Việt
                
                # Lấy thông tin chi tiết từ plant_info
                plant_details = plant_info.get(label_code, {})
                plant_description = plant_details.get("description", "Không có thông tin chi tiết.")
                plant_image_url = plant_image_urls.get(label_code, None)  # Lấy URL ảnh từ plant_image_urls

                # Thay thế phần trong ngoặc bằng in nghiêng
                italicized_description = italicize_latin_in_description(plant_description)

                with st.expander(f"{i + 1}. {plant_name_vietnamese} ({top_5_confidences[i].item():.2f}%)"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if plant_image_url:
                            img = load_image_from_url(plant_image_url)
                            if img:
                                st.image(img, caption=f"Hình ảnh của {plant_name_vietnamese}")
                    with col2:
                        st.subheader(plant_name_vietnamese)
                        # Hiển thị mô tả cây với phần trong ngoặc đã được in nghiêng
                        st.markdown(italicized_description)

    # Dòng cảm ơn
    st.markdown("---")
    st.markdown(
        """
        <div style='position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); width: auto; text-align: center; font-size: 10px; 
    font-family: "Arial", sans-serif; font-weight: color: white; font-style: italic;'>
            Design by NMai
        </div>
        """, 
        unsafe_allow_html=True
    )


elif page == "Trang đối chiếu":  # Đảm bảo dòng này không có vấn đề
    st.title("Thông tin Dược liệu (Tham khảo từ sách Dược liệu-Trường đại học Dược Hà Nội)")

    if labels and plant_info:
        # Sắp xếp tên cây theo bảng chữ cái
        vietnamese_labels = sorted([label_mapping.get(label, label) for label in labels])

        selected_plant = st.selectbox("Chọn cây để xem thông tin:", options=vietnamese_labels)

        # Lấy mã cây tương ứng từ tên cây đã chọn
        selected_label_code = next((k for k, v in label_mapping.items() if v == selected_plant), None)

        if selected_label_code:
            plant_details = plant_info.get(selected_label_code, {})
            plant_name = plant_details.get("name", "Không rõ")
            plant_description = plant_details.get("description", "Không có thông tin.")
            plant_image_url = plant_image_urls.get(selected_label_code, None)

            # Lấy tên Latin nếu có
            latin_name = plant_details.get("latin_name", None)  # Kiểm tra trường hợp không có latin_name

            col1, col2 = st.columns([1, 2])
            with col1:
                if plant_image_url:
                    img = load_image_from_url(plant_image_url)
                    if img:
                        st.image(img, caption=f"Hình ảnh {plant_name}")
            with col2:
                if latin_name:
                    # In nghiêng tên Latin nếu có
                    st.markdown(f"_{latin_name}_")  # Dùng st.markdown thay vì st.subheader
                st.subheader(plant_name)
                # Hiển thị mô tả cây với phần trong ngoặc đã được in nghiêng
                st.markdown(italicize_latin_in_description(plant_description))
