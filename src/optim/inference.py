import torch
import torch.nn as nn
import pandas as pd
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
            for imgs in pbar:
                imgs = imgs.float().to(device)
                pred = model(imgs)
                preds += pred.argmax(1).detach().cpu().numpy().tolist()
        
        preds = label_encoder.inverse_transform(preds)
    return preds

def make_submit(preds, sample_submit_file_path, submit_save_path):
    submit = pd.read_csv(sample_submit_file_path)
    submit['label'] = preds
    submit.to_csv(submit_save_path, index=False)