import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import Compose, PILToTensor, ConvertImageDtype

# Define the model architecture
def get_model(num_classes=2):
    # Load a pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the head of the classifier with a new one (adjust num_classes accordingly)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model()
model.load_state_dict(torch.load('./Models/Batch2/epoch_2_trained_model_litter_detection.pth'))
model.eval()

# Prepare the image
def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = Compose([
        PILToTensor(),
        ConvertImageDtype(torch.float32)
    ])
    image = transform(image)
    # Add a batch dimension
    return image

image = prepare_image('data/images/BATCH_d06_img_130.jpg')

# Perform object detection
with torch.no_grad():
    predictions = model([torch.rand(3,300,400)])

print(image.shape)
# Process predictions
boxes = predictions[0]['boxes']
scores = predictions[0]['scores']
labels = predictions[0]['labels']

print("Boxes:", boxes)
print("Scores:", scores)
print("Labels:", labels)