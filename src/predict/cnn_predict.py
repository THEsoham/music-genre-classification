import torch
import numpy as np
from src.preprocess.cnn_preprocess import preprocess_cnn

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

def predict_cnn(model, file_path):
    x = preprocess_cnn(file_path)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).squeeze().numpy()

    pred_idx = np.argmax(probs)

    return GENRES[pred_idx], {
        GENRES[i]: float(probs[i]) for i in range(len(GENRES))
    }