import torch
import torch.nn as nn
from torchvision import models

class CNN_Pretrained(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
