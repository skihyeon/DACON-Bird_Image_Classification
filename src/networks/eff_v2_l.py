import torch.nn as nn
import torchvision.models as models

class eff_v2_l(nn.Module):
    def __init__(self, le):
        super().__init__()
        num_clasees = len(le.classes_)
        self.backbone = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        self.classifier = nn.Linear(1000, num_clasees)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x