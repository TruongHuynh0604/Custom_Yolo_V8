import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("save_model.pt")

# Open camera (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from the camera.")
    exit()

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image from camera.")
        break

    # Perform object detection on the frame using YOLOv8
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
