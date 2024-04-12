import torch.nn as nn
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, le):
        super().__init__()
        num_clasees = len(le.classes_)
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.classifier = nn.Linear(1000, num_clasees)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x