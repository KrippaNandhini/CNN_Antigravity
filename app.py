
import os
import torch
import torchvision.transforms as transforms
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from src.model import create_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_model()
model.load_state_dict(torch.load('models/cifar10_cnn.pth', map_location=device))
model.to(device)
model.eval()

# Classes
CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0).to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Predict
            input_tensor = transform_image(file_path)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = CLASSES[predicted.item()]
            
            return render_template('index.html', prediction=predicted_class, image_path=file_path)
    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
