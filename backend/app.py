from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
import io
import torchvision.transforms as transforms
import torch
from ultralytics import YOLO

app = Flask(__name__)
# CORS(app,origins=['https://remarkable-lebkuchen-15f03a.netlify.app'])  # Enable CORS for all routes

CORS(app)
# Load your YOLO model
model = YOLO('best.pt')  # Load the model directly

# Preprocessing function (adapt as per your model input requirements)
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Adjust according to the model's input size
        transforms.ToTensor()
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Ensure image is in RGB format
    return transform(image).unsqueeze(0)  # Add batch dimension

@app.route('/')
def home():
    return "Welcome to the Road Accident Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    img_tensor = transform_image(img_bytes)
    
    with torch.no_grad():
        prediction = model(img_tensor)  # Perform inference
        
        # Extract prediction details
        if isinstance(prediction, list):
            prediction = prediction[0]  # YOLO returns a list of predictions

        result = {
            "boxes": prediction.boxes.xyxy.tolist(),  # Bounding boxes
            "scores": prediction.boxes.conf.tolist(),  # Confidence scores
            "classes": prediction.boxes.cls.tolist()    # Class indices
        }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
