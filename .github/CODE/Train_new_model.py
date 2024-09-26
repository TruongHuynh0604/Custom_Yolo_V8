from ultralytics import YOLO

# Tải mô hình YOLOv8 từ pre-trained model
model = YOLO('yolov8n.pt')  # Bạn có thể thay thế 'yolov8n.pt' bằng mô hình khác nếu cần

# Huấn luyện mô hình
results = model.train(
    data=r"D:\Project_Python\ultralytics-main\ultralytics-main\.github\CODE\Pin 3DC.v1i.yolov8\data.yaml",  # Đường dẫn đến tệp data.yaml
    epochs=10,                # Số epoch huấn luyện
    batch=16,                  # Kích thước batch
    imgsz=640,                 # Kích thước ảnh đầu vào (640x640)
    device='cpu'              # Sử dụng GPU, nếu không có GPU có thể dùng 'cpu'
)

# Kiểm tra mô hình
val_results = model.val()

# Lưu mô hình sau khi huấn luyện
model.save(r'D:\Project_Python\ultralytics-main\ultralytics-main\.github\CODE\save_model.pt')  # Đường dẫn để lưu mô hình

# In kết quả huấn luyện
print(results)
