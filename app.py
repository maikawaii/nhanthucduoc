# Trang chủ
if page == "Trang chủ":
    st.title("Nhận diện dược liệu")
    uploaded_file = st.file_uploader(
        "Nhập ảnh vào đây bạn nhé sau khi dự đoán vui lòng nhấn vào kết quả để xem thêm thông tin", 
        type=["jpg", "jpeg", "png"]
    )

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
            if st.button(f"{i + 1}. {label} ({confidence:.2f}%)"):
                st.session_state["selected_plant"] = label
                st.session_state["page"] = "Trang đối chiếu"
                st.experimental_rerun()

# Trang đối chiếu
if page == "Trang đối chiếu":
    st.title("Thông tin Dược liệu")

    # Kiểm tra cây được chọn từ trạng thái
    selected_plant = st.session_state.get("selected_plant", None)

    if selected_plant and labels and plant_info:
        # Hiển thị thông tin cây được chọn
        plant_details = plant_info.get(selected_plant)

        if plant_details:
            st.subheader(selected_plant)
            plant_details = plant_details.split("\n")

            # Hiển thị thông tin cây
            for detail in plant_details:
                st.write(detail)
        else:
            st.warning("Không có thông tin cho cây này.")
    else:
        st.warning("Vui lòng dự đoán và chọn một cây để xem thông tin.")
