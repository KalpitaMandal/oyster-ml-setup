from flask import Flask, jsonify
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# # Load Iris dataset from Sklearn
# iris = load_iris()
# x = iris.data
# y = iris.target

# # Split dataset
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# x_train = torch.FloatTensor(x_train)
# x_test = torch.FloatTensor(x_test)
# y_train = torch.LongTensor(y_train)
# y_test = torch.LongTensor(y_test)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Convolutional encoder
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel

        # Fully connected layers / Dense block
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.fc2 = nn.Linear(120, 84)         # 120 inputs, 84 outputs
        self.fc3 = nn.Linear(84, 10)          # 84 inputs, 10 outputs (number of classes)

    def forward(self, x):
        # Convolutional block
        x = F.avg_pool2d(F.sigmoid(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
        x = F.avg_pool2d(F.sigmoid(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

        # Flattening
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)  # No activation function here, will use CrossEntropyLoss later
        return x

app = Flask(__name__)

# Load trained model
model = LeNet()
model.load_state_dict(torch.load('/app/mnist_97.pth'))
model.eval()

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # data = request.get_json(force=True)
    f = open('/app/input.json')
    data = json.load(f)
    features = data['input_data']
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    with torch.no_grad():
        output = model(features_tensor)
        _, predicted_class = torch.max(output, 1)
    prediction = predicted_class.item()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
