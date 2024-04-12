import numpy as np
import torch
import copy
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader, 
                 optimizer: Optimizer, 
                 loss_func: nn.Module,  
                 num_epochs: int,
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
        self.device = device
        self.save_path = save_path
        self.loss_func = loss_func.to(device)
        self.model.to(device)
        self.wandb = wandb

    def train(self):
        best_score = 0
        best_model = None

        with tqdm(range(self.num_epochs), desc='Epochs') as pbar:
            for epoch in pbar:
                self.model.train()
                # train_loss = []
                train_loss_sum = 0  # Initialize sum of losses for each epoch
                num_batches = 0  # Count the number of batches processed
                for imgs, labels in tqdm(self.train_loader, desc="Iter"):
                    imgs, labels = imgs.float().to(self.device), labels.to(self.device)
                    labels = labels.long()

                    self.optimizer.zero_grad()
                    output = self.model(imgs)
                    loss = self.loss_func(output, labels)

                    loss.backward()
                    self.optimizer.step()

                    # train_loss.append(loss.item())
                    train_loss_sum += loss.detach()  # Accumulate loss on GPU
                    num_batches += 1

                val_loss, val_score = self.validate()
                # train_loss = np.mean(train_loss)
                train_loss_avg = train_loss_sum / num_batches
                pbar.set_postfix_str(f'Epoch [{epoch}], Train Loss: {train_loss_avg:.5f}, Val Loss: {val_loss:.5f}, Val F1 Score: {val_score:.5f}')

                if self.wandb is not None:
                    self.wandb.log({'train/epoch':epoch, 
                                    'train/loss':train_loss_avg.item(), 
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
                    torch.jit.save(scripted_best_model, self.save_path)
                    print(f"\nscore {best_score:.3f} model saved to {self.save_path}")

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