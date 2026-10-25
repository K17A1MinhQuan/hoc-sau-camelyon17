"""model.py — Xây dựng mô hình ResNet-18 cho phân loại nhị phân."""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_model(num_classes=2):
    """ResNet-18 pretrained ImageNet, thay lớp FC cuối cho binary classification."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
