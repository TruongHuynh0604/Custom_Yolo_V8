from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

# Tải mô hình YOLOv8 đã lưu
model_path = r'D:\Project_Python\ultralytics-main\ultralytics-main\.github\CODE\save_model.pt'
model = YOLO(model_path)

# Đường dẫn đến thư mục chứa ảnh
input_folder = r'D:\Project_Python\ultralytics-main\ultralytics-main\.github\CODE\Image\Image_Change_640'
output_folder = r'D:\Project_Python\ultralytics-main\ultralytics-main\.github\CODE\output'

# Đảm bảo thư mục đầu ra tồn tại
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lấy danh sách tất cả các tệp ảnh trong thư mục đầu vào
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Xử lý từng ảnh trong danh sách
for i, image_file in enumerate(image_files):
    # Đường dẫn đầy đủ đến ảnh
    image_path = os.path.join(input_folder, image_file)

    # Dự đoán trên ảnh
    results = model.predict(source=image_path)

    # Lưu từng kết quả dự đoán
    for j, result in enumerate(results):
        # Tạo tên tệp kết quả
        output_path = os.path.join(output_folder, f'{os.path.splitext(image_file)[0]}_result_{j}.jpg')

        # Lưu hình ảnh dự đoán
        result.plot(save=True, filename=output_path)

print("Dự đoán hoàn tất và kết quả đã được lưu tại:", output_folder)
