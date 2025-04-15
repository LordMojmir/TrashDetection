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


# Load JSON data function
def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def load_split_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['train'], data['val'], data['test']

# Define custom dataset for litter detection
class LitterDataset(Dataset):
    def __init__(self, data, root_dir, max_boxes=10):
        self.data = data
        self.root_dir = root_dir
        self.max_boxes = max_boxes
        self.transforms = Compose([PILToTensor(), ConvertImageDtype(torch.float32)])

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img_data = next((item for item in self.data['images'] if item['id'] == idx), None)
        if img_data is None:
            raise ValueError("Image ID not found")
        img_path = os.path.join(self.root_dir, img_data['file_name'])
        image = Image.open(img_path).convert("RGB")

        annotations = [ann for ann in self.data['annotations'] if ann['image_id'] == idx]
        boxes = [[ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]] for
                 ann in annotations]
        # labels = [0] * len(boxes)  # Assuming 'rubbish' is labeled as 0

        labels = torch.ones((len(boxes),), dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.as_tensor(labels, dtype=torch.int64)

        image = self.transforms(image)

        return image, {"boxes": boxes, "labels": labels}

# Load the data
data = load_data('data/annotations.json')
dataset = LitterDataset(data, 'data/images')
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Initialize the model
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 2  # Including background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer and learning rate scheduler setup
# optimizer = optim.Adam(model.parameters()) # lr=0.005 or lr=0.025, momentum=0.9, weight_decay=0.0005
optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# model.load_state_dict(torch.load('trained_model_litter_detection.pth'))

# Training loop
num_epochs = 10
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in data_loader:
        # print("target", targets)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # print(targets)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}")
    torch.save(model.state_dict(), f"Models/Batch3/epoch_{epoch + 1}_trained_model_litter_detection.pth")
    lr_scheduler.step()

print("Training finished!")

model.load_state_dict(torch.load('Models/epoch_1.0_trained_model_litter_detection.pth'))
model.eval()
# Image transformations
transforms = Compose([
    PILToTensor(),
    ConvertImageDtype(torch.float32)
])

# Load and transform the image
# image_path = 'data/images/BATCH_d06_img_130.jpg'
image_path = 'data/real/Drone_V2.png'
image = Image.open(image_path).convert("RGB")
image = transforms(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
print(image.min(), image.max())

# Perform inference
with torch.no_grad():
    result = model(image)

    # Output the result
    print(result)

    # Visualize the result
    fig, ax = plt.subplots(1)
    ax.imshow(image.squeeze(0).cpu().numpy().transpose(1, 2, 0))
    for box in result[0]['boxes']:
        box = box.cpu().numpy()
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()