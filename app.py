from flask import Flask, render_template, request, redirect, url_for
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import os

app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data directory and model path
model_path = "./project/savedmodel/resnet50_model.pth"

# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Define function to rename keys
def rename_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "fc.1" in key:
            new_key = key.replace("fc.1", "fc")
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

# Load state dict with map_location to properly place tensors on the given device
state_dict = torch.load(model_path, map_location=device)
renamed_state_dict = rename_keys(state_dict)
model.load_state_dict(renamed_state_dict, strict=False)  # Use strict=False to ignore missing and unexpected keys
model = model.to(device)
model.eval()  # Set model to evaluation mode

def predict_image(image_bytes, model, transform):
    try:
        print("Attempting to open image...")
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert image to RGB if it's not
        if image.mode != "RGB":
            print("Converting image to RGB...")
            image = image.convert("RGB")
        
        print("Transforming image...")
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        print("Performing prediction...")
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = 'Normal' if predicted.item() == 0 else 'Pneumonia'
        
        print(f"Predicted class: {predicted_class}")
        return predicted_class
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            image_bytes = file.read()
            prediction = predict_image(image_bytes, model, data_transforms)

            if prediction is not None:
                return render_template('result.html', prediction=prediction)
            else:
                return "Prediction failed. Please try again."

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
