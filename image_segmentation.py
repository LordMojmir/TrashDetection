import os
import json
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import Compose, PILToTensor, ConvertImageDtype
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure matplotlib uses TkAgg as the backend
import matplotlib
matplotlib.use('TkAgg')

# Initialize the model
model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 2  # Including background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load model weights
model.load_state_dict(torch.load('./Models/Batch2/epoch_6_trained_model_litter_detection.pth'))
model.eval()

# Image transformations
transforms = Compose([
    PILToTensor(),
    ConvertImageDtype(torch.float32)
])

def process_image_segment(image):
    """ Process a single image segment through the model """
    image_tensor = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        result = model(image_tensor)
    return result, image_tensor

def segment_image(image, max_width=512, max_height=256):
    """ Segment the image if it is larger than max_width x max_height """
    width, height = image.size
    segments = []
    for i in range(0, height, max_height):
        for j in range(0, width, max_width):
            box = (j, i, min(j+max_width, width), min(i+max_height, height))
            segment = image.crop(box)
            segments.append(segment)
    return segments

# Load and segment the image
image_path = 'data/real/Drone_Sunday.png'
original_image = Image.open(image_path).convert("RGB")
segments = segment_image(original_image)

# Create one big image for combined visualization
total_height = sum(seg.size[1] for seg in segments)
composite_image = Image.new('RGB', (max(seg.size[0] for seg in segments), total_height))
current_height = 0

for segment in segments:
    composite_image.paste(segment, (0, current_height))
    current_height += segment.size[1]

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(composite_image)

current_height = 0
for segment in segments:
    result, _ = process_image_segment(segment)
    scores = result[0]['scores'].cpu().numpy()
    boxes = result[0]['boxes'].cpu().numpy()
    for score, box in zip(scores, boxes):
        print("Score", score)
        if score > 0.4:
            adjusted_box = [box[0], box[1] + current_height, box[2], box[3] + current_height]
            rect = patches.Rectangle((adjusted_box[0], adjusted_box[1]), adjusted_box[2] - adjusted_box[0], adjusted_box[3] - adjusted_box[1],
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    current_height += segment.size[1]

ax.set_axis_off()
plt.show()
