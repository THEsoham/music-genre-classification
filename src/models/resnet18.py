import torch.nn as nn
from torchvision import models

def get_resnet_model():
    model = models.resnet18(pretrained=False)  # 🔥 IMPORTANT
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model