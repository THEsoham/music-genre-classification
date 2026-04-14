from flask import Flask, request, jsonify, render_template
import torch
import os

# CNN
from src.models.cnn import AudioCNN
from src.predict.cnn_predict import predict_cnn

# ResNet
from src.models.resnet18 import get_resnet_model
from src.predict.resnet_predict import predict_resnet

app = Flask(__name__, template_folder="templates", static_folder="static")

# -----------------------------
# PATH SETUP (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cnn_path = os.path.join(BASE_DIR, "weights", "cnn_model.pth")
resnet_path = os.path.join(BASE_DIR, "weights", "resnet_model.pth")

# -----------------------------
# CHECK FILES EXIST
# -----------------------------
if not os.path.exists(cnn_path):
    raise FileNotFoundError(f"Missing file: {cnn_path}")

if not os.path.exists(resnet_path):
    raise FileNotFoundError(f"Missing file: {resnet_path}")

# -----------------------------
# LOAD MODELS
# -----------------------------
cnn_model = AudioCNN()
cnn_model.load_state_dict(torch.load(cnn_path, map_location="cpu"))
cnn_model.eval()

resnet_model = get_resnet_model()
resnet_model.load_state_dict(torch.load(resnet_path, map_location="cpu"))
resnet_model.eval()

print("✅ Models loaded successfully")

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Save temp file safely
        temp_path = os.path.join(BASE_DIR, "temp.wav")
        file.save(temp_path)

        # Predictions
        cnn_label, cnn_probs = predict_cnn(cnn_model, temp_path)
        res_label, res_probs = predict_resnet(resnet_model, temp_path)

        return jsonify({
            "cnn": {
                "prediction": cnn_label,
                "probabilities": cnn_probs
            },
            "resnet": {
                "prediction": res_label,
                "probabilities": res_probs
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# LOCAL RUN (ONLY FOR DEBUG)
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
