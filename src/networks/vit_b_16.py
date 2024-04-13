import torch.nn as nn
import torchvision.models as models
from  torchvision.models import ViT_B_16_Weights

class vit_b_16(nn.Module):
    def __init__(self, le):
        super().__init__()
        num_classes = len(le.classes_)
        self.backbone = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
