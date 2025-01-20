from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import logging
from flask_cors import CORS

logging.basicConfig(level=logging.DEBUG)

class Model(nn.Module):
    def __init__(self, in_features=7, h1=16, h2=16, out_features=22):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

model = Model()
model.load_state_dict(torch.load('crop_prediction_model.pt'))
model.eval()

label_encoder = joblib.load("label_encoder.pkl")

app = Flask(__name__)
# Allow all origins, methods and headers
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],
        "allow_headers": "*",
        "expose_headers": "*",
        "supports_credentials": True
    }
})
@app.route('/')
def home():
    return "Welcome to the Crop Prediction API"

@app.route("/test", methods=['GET'])
def test():
    logger.info("Test endpoint accessed")
    return jsonify({"status": "API is working"})

@app.route("/predict", methods=['POST'])
def predict():
    app.logger.debug(f"Request JSON: {request.json}")
    try:
        n = float(request.json.get('N'))
        p = float(request.json.get('P'))
        k = float(request.json.get('K'))
        temp = float(request.json.get('temperature'))
        humidity = float(request.json.get('humidity'))
        ph = float(request.json.get('ph'))
        rainfall = float(request.json.get('rainfall'))
    except (TypeError, ValueError) as e:
        return jsonify({"error": "Invalid input data", "details": str(e)}), 400
    input_query = torch.tensor([[n, p, k, temp, humidity, ph, rainfall]])

    output = model(input_query)

    predicted_class = torch.argmax(output, dim=1).item()


    predicted_class_name = label_encoder.inverse_transform([predicted_class])[0]

    return jsonify({
        "class_name": predicted_class_name
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# simply run: python app.py