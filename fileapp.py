from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from io import BytesIO

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define image transformations
transform = Compose([
    Resize((416, 416)),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define route for object detection API
@app.route('/detect', methods=['POST'])
def detect():
    # Get image from request
    file = request.files['image']
    image_bytes = file.read()

    # Preprocess image
    image = Image.open(BytesIO(image_bytes))
    input_tensor = transform(image).unsqueeze(0)

    # Perform object detection
    results = model(input_tensor)

    # Convert results to JSON format
    output = [{'class': int(class_id), 'name': model.names[int(class_id)], 'confidence': float(confidence),
               'xmin': int(xmin), 'ymin': int(ymin), 'xmax': int(xmax), 'ymax': int(ymax)}
              for class_id, confidence, xmin, ymin, xmax, ymax in results.xyxy[0].tolist()]

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
