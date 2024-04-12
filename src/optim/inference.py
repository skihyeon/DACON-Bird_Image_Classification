import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def inference(model: nn.Module, 
              test_loader: DataLoader, 
              label_encoder,
              device: str):
    model.eval()
    preds = []
    with torch.no_grad():
        with tqdm(iter(test_loader), desc='Inference') as pbar:
            for imgs in tqdm(iter(test_loader)):
                imgs = imgs.float().to(device)
                pred = model(imgs)
                preds += pred.argmax(1).detach().cpu().numpy().tolist()
        
        preds = label_encoder.inverse_transform(preds)
    return preds