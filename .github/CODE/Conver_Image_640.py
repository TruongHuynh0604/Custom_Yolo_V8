import cv2
import os
import numpy as np

# Đường dẫn đến thư mục chứa ảnh gốc
input_folder = "D:/Project_Python/ultralytics-main/ultralytics-main/.github/CODE/Image/Image_4M"
# Đường dẫn đến thư mục lưu ảnh đã thay đổi kích thước
output_folder = "D:/Project_Python/ultralytics-main/ultralytics-main/.github/CODE/Image/Image_Change_640"

# Tạo thư mục lưu ảnh mới nếu chưa tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Duyệt qua tất cả các tệp trong thư mục
for filename in os.listdir(input_folder):
    # Chỉ xử lý các tệp có định dạng ảnh (ví dụ: .jpg, .png)
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # Đọc đường dẫn đầy đủ của tệp
        img_path = os.path.join(input_folder, filename)

        # Đọc hình ảnh
        image = cv2.imread(img_path)

        # Kiểm tra xem hình ảnh có được đọc thành công không
        if image is None:
            print(f"Error: Could not read image {filename}")
            continue

        # Lấy kích thước của hình ảnh gốc
        h, w, _ = image.shape

        # Tính toán tỉ lệ để thay đổi kích thước mà vẫn giữ nguyên tỷ lệ ảnh
        scale = 640 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Thay đổi kích thước hình ảnh giữ nguyên tỷ lệ
        resized_image = cv2.resize(image, (new_w, new_h))

        # Tạo viền đen (padding) xung quanh để đưa ảnh về kích thước 640x640
        top = (640 - new_h) // 2
        bottom = 640 - new_h - top
        left = (640 - new_w) // 2
        right = 640 - new_w - left

        # Thêm viền vào hình ảnh
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Đường dẫn lưu ảnh đã xử lý
        save_path = os.path.join(output_folder, filename)

        # Lưu lại hình ảnh đã thêm viền
        cv2.imwrite(save_path, padded_image)

        print(f"Image {filename} resized to 640x640 with padding and saved successfully.")

print("All images processed.")
