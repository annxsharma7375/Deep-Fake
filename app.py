# import necessary libraries
from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from collections import OrderedDict

# initialize the Flask application
app = Flask(__name__)

# define a custom model class that extends nn.Module
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.base_model.fc.in_features
        # update the final layer to output 2 classes (Real, Deep Fake)
        self.base_model.fc = nn.Linear(num_ftrs, 2)

    # forward pass definition
    def forward(self, x):
        return self.base_model(x)

# path to the saved model weights
MODEL_PATH = "model.pth"
# set the device to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel()  # Initialize the custom model

# load the saved model weights from a file
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)

# adjust the state dict to match the model's architecture
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # update the keys to match the model's architecture
    new_key = f"base_model.{k}" 
    new_state_dict[new_key] = v

# load the modified state_dict into the model
model.load_state_dict(new_state_dict)
model = model.to(device)  # move the model to the selected device (GPU/CPU)
model.eval()  # set the model to evaluation mode

# define the image transformation pipeline for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize the image to 224x224 (required by ResNet)
    transforms.ToTensor(),  # convert image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalize based on ImageNet statistics
])

# route to render the index (homepage) with the form for uploading images
@app.route('/')
def index():
    return render_template('template-1.html')

# route to handle image prediction when the user uploads a file
@app.route('/predict', methods=['POST'])
def predict():
    # check if a file is provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    
    # check if a file was actually selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # open the image file and convert it to RGB
        img = Image.open(file).convert('RGB')
        
        # apply the transformation (resize, normalize, etc.)
        img_tensor = transform(img).unsqueeze(0).to(device)

        # run the model on the transformed image
        with torch.no_grad():
            outputs = model(img_tensor)
            # get the predicted class (Real or Deep Fake)
            _, predicted = torch.max(outputs, 1)
            label = 'Real' if predicted.item() == 1 else 'Deep Fake'

        # return the prediction as a JSON response
        return jsonify({'prediction': label})
    except Exception as e:
        # handle any error that occurs during processing
        return jsonify({'error': str(e)})

# start the Flask application
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")