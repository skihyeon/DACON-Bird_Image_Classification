import torch
import os
import pickle
import importlib

from utils.utils import seed_everything, wandb_login, update_wandb_config, resume_latest_run_log
from optim.trainer import BaseTrainer
from optim.inference import inference, make_submit
from optim.sam import SAM
from datasets.dataloader import label_preprocessing, get_train_loader, get_val_loader, get_test_loader
from configs.config import Config

class ModelFactory:
    @staticmethod
    def get_model(model_name, label_encoder):
        module = importlib.import_module(f"networks.{model_name}")
        model_class = getattr(module, model_name)
        return model_class(label_encoder)


def train_func(run_name, model_name, exp_path, 
          project_name, seed, batch_size, 
          img_resize_size, shuffle, train_csv_path, 
          wandb_logging, wandb_account_entity, keep_train, keep_train_model_file, 
          test_split_ratio, lr, num_epochs, epochs_per_save):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config(locals().copy())
    seed_everything(config.settings['seed'])
    log_path = os.path.normpath(exp_path + run_name + '/') + '/'

    if model_name is None:
        ValueError("Please choose model!")

    if not os.path.exists(log_path):
        assert keep_train is False, 'There is no trained model to resume!'
        os.makedirs(log_path)
    
    wandb = None
    config.save_config(log_path+'config.json')
    if wandb_logging is True:
        wandb = wandb_login()
        if keep_train is True:
             resume_latest_run_log(wandb,
                                   entity=config.settings['wandb_account_entity'],
                                   project_name=config.settings['project_name'],
                                   target_run_name=config.settings['run_name']
                                   )
        else:
            wandb.init(project=config.settings['project_name'], 
                       entity=config.settings['wandb_account_entity'], 
                       reinit=True, 
                       name=config.settings['run_name'])
        update_wandb_config(wandb, config)
    
    label_encoder, train_df, val_df = label_preprocessing(config.settings['train_csv_path'], config.settings['test_split_ratio'])
    
    if keep_train and os.path.exists(log_path+'label_encoder.pkl'):
        with open(log_path+'label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)
    else:
        with open(log_path + 'label_encoder.pkl', 'wb') as file:
            pickle.dump(label_encoder, file)

    train_loader = get_train_loader(train_df, config.settings['img_resize_size'],
                                    config.settings['batch_size'], config.settings['shuffle'])
    val_loader = get_val_loader(val_df, config.settings['img_resize_size'],
                                config.settings['batch_size'], config.settings['shuffle'])
    model = ModelFactory.get_model(model_name, label_encoder).to(device)
    base_optimizer = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_optimizer, lr=config.settings['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.base_optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8)

    loss_func = torch.nn.CrossEntropyLoss()
    Trainer = BaseTrainer(model=model,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          optimizer=optimizer,
                          loss_func=loss_func,
                          num_epochs=config.settings['num_epochs'],
                          epochs_per_save = epochs_per_save,
                          device=device,
                          save_path= log_path,
                          scheduler=scheduler,
                          wandb=wandb)
    keep_train_model_path = log_path + keep_train_model_file if keep_train is True else None
    best_model = Trainer.train(keep_train, keep_train_model_path)


def inference_func(run_name, model_name, exp_path, 
              project_name, seed, batch_size, 
              img_resize_size, shuffle, test_csv_path,
              load_model, sample_submit_file_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config(locals().copy())
    seed_everything(config.settings['seed'])
    log_path = os.path.normpath(exp_path + run_name + '/') + '/'

    with open(log_path+'label_encoder.pkl', 'rb') as file:
                label_encoder = pickle.load(file)

    test_loader = get_test_loader(config.settings['test_csv_path'],
                                  config.settings['img_resize_size'],
                                  config.settings['batch_size'],
                                  config.settings['shuffle']) 
    
    if load_model is None:
        ValueError("Model Path Error!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelFactory.get_model(model_name, label_encoder)
    model.load_state_dict(torch.load(log_path + load_model, map_location='cpu'))
    model.to(device)  # 모델 상태를 로드한 후 필요한 디바이스로 이동
    model.eval()
    preds = inference(model, test_loader, label_encoder, device)
    if os.path.exists(sample_submit_file_path):
                model_file_name = config.settings['load_model'].replace('.pt', "")
                submit_save_path = log_path + f'submit_{run_name}_{model_file_name}.csv'
                make_submit(preds, sample_submit_file_path, submit_save_path)
    else:
        print("Sample submit files is not exists!")
