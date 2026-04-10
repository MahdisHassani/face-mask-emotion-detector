import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.shared_dim = 1280

        # mask head
        self.mask_head = nn.Sequential(
            nn.Linear(self.shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # emotion head
        self.emotion_head = nn.Sequential(
            nn.Linear(self.shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        mask_out = self.mask_head(x)
        emotion_out = self.emotion_head(x)

        return mask_out, emotion_out