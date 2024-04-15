import numpy as np
import torch
import copy
import os

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import re 

class BaseTrainer:
    def __init__(self, model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader, 
                 optimizer: Optimizer, 
                 loss_func: nn.Module,  
                 num_epochs: int,
                 epochs_per_save: int,
                 device: str,
                 save_path: str,
                 scheduler = None,
                 wandb = None,
                 ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.epochs_per_save = epochs_per_save
        self.device = device
        self.save_path = save_path
        self.loss_func = loss_func.to(device)
        self.model.to(device)
        self.wandb = wandb

    def train(self, keep_train=False, keep_train_model_file=None):
        start_epoch = 0
        if keep_train:
            start_epoch = self.load_model(keep_train_model_file)
        best_score = 0
        best_model = None

        with tqdm(range(start_epoch, start_epoch + self.num_epochs), desc='Epochs') as pbar:
            for epoch in pbar:
                self.model.train()
                train_loss_sum = 0
                num_batches = 0

                with tqdm(self.train_loader, desc="Iter") as batch_bar:
                    for imgs, labels in batch_bar:
                        imgs, labels = imgs.float().to(self.device), labels.to(self.device)
                        labels = labels.long()

                        self.optimizer.zero_grad()
                        output = self.model(imgs)
                        loss = self.loss_func(output, labels)

                        loss.backward()
                        self.optimizer.step()

                        train_loss_sum += loss.item()
                        num_batches += 1

                        current_loss_avg = train_loss_sum / num_batches
                        batch_bar.set_postfix(loss=current_loss_avg)

                        self.wandb.log({'train/iter': current_loss_avg})
                            

                val_loss, val_score = self.validate()
                train_loss_avg = train_loss_sum / num_batches
                print(f'Epoch [{epoch}], Train Loss: {train_loss_avg:.3f}, Val Loss: {val_loss:.3f}, Val F1 Score: {val_score:.3f}')

                if epoch % self.epochs_per_save == 0:  
                    filename = f"model_{epoch}.pt"
                    save_path = f"{self.save_path}/{filename}"
                    scripted_model = torch.jit.script(self.model)
                    torch.jit.save(scripted_model, save_path)
                    print(f"Epoch:{epoch} Model saved to {save_path}")

                if self.wandb is not None:
                    self.wandb.log({'train/loss':train_loss_avg, 
                                    'val/loss':val_loss, 
                                    'val/score':val_score,
                                    })

                if self.scheduler is not None:
                    self.scheduler.step(val_score)
                    if self.wandb is not None:
                        self.wandb.log({'train/learning_rate': self.scheduler.get_last_lr()[0]})

                if best_score < val_score:
                    best_score = val_score
                    best_model = copy.deepcopy(self.model)
                    scripted_best_model = torch.jit.script(best_model)
                    torch.jit.save(scripted_best_model, f"{self.save_path}/best_model_ep{epoch}.pt")


        return best_model

    def validate(self):
        self.model.eval()
        val_loss = []
        preds, true_labels = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc='Val'):
                imgs, labels = imgs.float().to(self.device), labels.to(self.device)
                labels = labels.long()

                pred = self.model(imgs)
                loss = self.loss_func(pred, labels)

                preds.extend(pred.argmax(1).detach().cpu().numpy().tolist())
                true_labels.extend(labels.detach().cpu().numpy().tolist())

                val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
        val_score = f1_score(true_labels, preds, average='macro')
        return val_loss, val_score

    def load_model(self, load_path=None):
        if load_path:
            if not os.path.exists(load_path):
                raise ValueError(f"The provided filename {load_path} does not exist")
            epoch = int(re.search(r"model_(\d+).pt", load_path).group(1))
            model = torch.jit.load(load_path, map_location=self.device)
            self.model.load_state_dict(model.state_dict())
            print(f"Model loaded from {load_path}, Resuming from epoch {epoch+1}")
            return epoch + 1
        return 0
    
    