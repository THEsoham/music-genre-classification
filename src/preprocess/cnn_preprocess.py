import librosa
import numpy as np
import torch

def preprocess_cnn(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] < 1300:
        pad = 1300 - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad)))
    else:
        mel = mel[:, :1300]

    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    return torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)