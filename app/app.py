from flask import Flask, request, jsonify, render_template
import torch

# CNN
from src.models.cnn import AudioCNN
from src.predict.cnn_predict import predict_cnn

# ResNet
from src.models.resnet18 import get_resnet_model
from src.predict.resnet_predict import predict_resnet

app = Flask(__name__)

# -----------------------------
# LOAD MODELS
# -----------------------------
cnn_model = AudioCNN()
cnn_model.load_state_dict(torch.load("weights/cnn_model.pth", map_location="cpu"))
cnn_model.eval()

resnet_model = get_resnet_model()
resnet_model.load_state_dict(torch.load("weights/resnet_model.pth", map_location="cpu"))
resnet_model.eval()

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    file_path = "temp.wav"
    file.save(file_path)

    cnn_label, cnn_probs = predict_cnn(cnn_model, file_path)
    res_label, res_probs = predict_resnet(resnet_model, file_path)

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


if __name__ == "__main__":
    app.run(debug=True)