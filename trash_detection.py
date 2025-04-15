import os
import json
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import Compose, PILToTensor, ConvertImageDtype
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure matplotlib uses TkAgg as the backend
import matplotlib

matplotlib.use('TkAgg')

# Initialize the model
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 2  # Including background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


model.load_state_dict(torch.load('./Models/Batch2/epoch_6_trained_model_litter_detection.pth'))
model.eval()
# Image transformations
transforms = Compose([
    PILToTensor(),
    ConvertImageDtype(torch.float32)
])

# Load and transform the image
# image_path = 'data/images/BATCH_d06_img_130.jpg'
image_path = 'data/real/Segment_2.png'
# image_path = 'data/real/Drone_Sunday.png'
image = Image.open(image_path).convert("RGB")
image = transforms(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
print(image.min(), image.max())
threshold = 0.25

# Perform inference
with torch.no_grad():
    result = model(image)

    # Output the result
    print(result)

    # Visualize the result
    # Visualize the result
    fig, ax = plt.subplots(1)
    ax.imshow(image.squeeze(0).cpu().numpy().transpose(1, 2, 0))
    scores = result[0]['scores'].cpu().numpy()
    boxes = result[0]['boxes'].cpu().numpy()

    # Iterate over the scores and corresponding boxes
    for score, box in zip(scores, boxes):
        if score > threshold:  # Use '>' to keep high-confidence predictions
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()
