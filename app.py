from flask import Flask, request, jsonify
import torch
from model import FakeNewsGNN  # Import your GNN model
from feature_extraction import extract_features  # Import your BERT feature extractor

app = Flask(__name__)

# Load model
model = FakeNewsGNN()  # Make sure this initializes correctly
model.load_state_dict(torch.load("fakenews_model.pth"))  # Adjust path if needed
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Extract BERT features
        features = extract_features(text)
        features = torch.tensor(features).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(features)
            prediction = torch.argmax(output, dim=1).item()

        result = "Fake" if prediction == 1 else "Real"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
