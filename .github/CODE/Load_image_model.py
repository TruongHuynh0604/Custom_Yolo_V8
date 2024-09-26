from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

# Tải mô hình YOLOv8 đã lưu
model_path = r'D:\Project_Python\ultralytics-main\ultralytics-main\.github\CODE\save_model.pt'
model = YOLO(model_path)

# Đường dẫn đến ảnh mới
image_path = r'D:\Project_Python\ultralytics-main\ultralytics-main\.github\CODE\Image\Image_Change_640\OK_ (82).jpg'

# Dự đoán trên ảnh mới
results = model.predict(source=image_path)

# Lưu kết quả dự đoán
save_dir = r'D:\Project_Python\ultralytics-main\ultralytics-main\.github\CODE\output'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Lưu từng kết quả dự đoán nếu cần
for i, result in enumerate(results):
    # Tạo tên tệp kết quả
    output_path = os.path.join(save_dir, f'result_{i}.jpg')

    # Lưu hình ảnh dự đoán
    result.plot(save=True, filename=output_path)

print("Dự đoán hoàn tất và kết quả đã được lưu tại:", save_dir)
