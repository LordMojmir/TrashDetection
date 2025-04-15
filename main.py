import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import Compose, PILToTensor, ConvertImageDtype
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Function to load the trained Faster R-CNN model
def load_model(model_path, num_classes=2):
    # Load the pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the head with a new one (correct number of classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


# Function to prepare the image
def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = Compose([PILToTensor(), ConvertImageDtype(torch.float32)])
    image = transform(image)
    return image


# Function to perform object detection
def get_predictions(model, image):
    with torch.no_grad():
        predictions = model([image])  # The model expects a list of images
    return predictions[0]


# Function to plot the image and its bounding boxes
def plot_image_with_boxes(image, predictions):
    image = image.permute(1, 2, 0).numpy()  # Convert CHW to HWC
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    boxes = predictions['boxes']
    scores = predictions['scores']
    for box, score in zip(boxes, scores):
        if score > 0.5:  # Threshold can be adjusted
            x, y, x2, y2 = box
            rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()


# Path to the trained model and image
model_path = 'trained_model_litter_detection.pth'
image_path = 'data/images/BATCH_d06_img_130.jpg'

# Load the model
model = load_model(model_path)

# Prepare the image
image = prepare_image(image_path)

# Get predictions
predictions = get_predictions(model, image)

# Plot the image with bounding boxes
plot_image_with_boxes(image, predictions)
