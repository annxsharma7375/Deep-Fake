from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from collections import OrderedDict

app = Flask(__name__)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.base_model(x)


MODEL_PATH = "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel()

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = f"base_model.{k}" 
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('template-1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        img = Image.open(file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            label = 'Real' if predicted.item() == 1 else 'Deep Fake'

        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
