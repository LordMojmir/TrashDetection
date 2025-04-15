from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10l.pt")


# Perform object detection on an imag   e
# results = model("litter_v1.png")
results = model("Drone_V1.png")

# Display the results
results[0].show()