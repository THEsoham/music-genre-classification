import torch
import numpy as np
from src.preprocess.resnet_preprocess import preprocess_resnet
from src.predict.cnn_predict import GENRES

def predict_resnet(model, file_path):
    x = preprocess_resnet(file_path)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).squeeze().numpy()

    pred_idx = np.argmax(probs)

    return GENRES[pred_idx], {
        GENRES[i]: float(probs[i]) for i in range(len(GENRES))
    }