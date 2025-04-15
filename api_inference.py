from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import Compose, PILToTensor, ConvertImageDtype
import io

app = Flask(__name__)
CORS(app)  # Enable CORS on all routes

# Define the model architecture
def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


model = get_model()
model.load_state_dict(torch.load('./Models/86_trained_model_litter_detection.pth'))
model.eval()

# Prepare the image from in-memory bytes
def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = Compose([
        PILToTensor(),
        ConvertImageDtype(torch.float32)
    ])
    return transform(image)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    print(file.filename)
    if file.filename == '':
        return "No selected file", 400
    if file:
        image = prepare_image(file.read())
        with torch.no_grad():
            predictions = model([image])

        # Example: Simply returning the first predicted box as an image (for demonstration)
        # Normally you'd process predictions further to generate meaningful output
        box = predictions[0]['boxes'][0].int().tolist()
        print(box)
        cropped_image = Image.fromarray(image.mul(255).byte().numpy().transpose(1, 2, 0)[box[1]:box[3], box[0]:box[2], :])
        byte_io = io.BytesIO()
        cropped_image.save(byte_io, 'PNG')
        byte_io.seek(0)
        return send_file(byte_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000) # , host='0.0.0.0'
