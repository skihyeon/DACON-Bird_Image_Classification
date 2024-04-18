from transformers import Swinv2Model
import torch.nn as nn

class swin_v2(nn.Module):
    def __init__(self, le):
        super().__init__()
        num_classes = len(le.classes_)
        self.backbone = Swinv2Model.from_pretrained("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft")
        self.classifier = nn.Sequential(
                                         nn.Tanh(),
                                         nn.LazyLinear(num_classes))

    def forward(self, x):
        x = self.backbone(x).pooler_output
        x = self.classifier(x)
        return x

