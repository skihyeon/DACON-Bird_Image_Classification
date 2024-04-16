import torch.nn as nn
import torchvision.models as models

class eff_b7(nn.Module):
    def __init__(self, le):
        super().__init__()
        num_clasees = len(le.classes_)
        self.backbone = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        self.classifier = nn.Linear(1000, num_clasees)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x