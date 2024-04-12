import torch
import click
import os
import pickle

from networks.basemodel import BaseModel
from utils.utils import seed_everything, wandb_login, update_wandb_config
from optim.trainer import BaseTrainer
from optim.inference import inference, make_submit
from datasets.dataloader import label_preprocessing, get_train_loader, get_val_loader, get_test_loader
from configs.config import Config


@click.command()
@click.argument('mode', type=click.Choice(['train', 'inference']))
@click.argument('run_name', type=str, default=None)
@click.option('--exp_path', type=click.Path(exists=True), default='../exps/')
@click.option('--train_csv_path', type=click.Path(exists=True), default='../datas/train.csv')
@click.option('--test_csv_path', type=click.Path(exists=True), default='../datas/test.csv')
@click.option('--project_name', type=str, default='low_res_bird_img_classification')
@click.option('--wandb_logging', type=bool, default=True)
@click.option('--wandb_account_entity', type=str, default='hero981001')
@click.option('--keep_train', type=bool, default=False)
@click.option('--keep_train_model_file', type=str, default=None)
@click.option('--seed', type=int, default=456)
@click.option('--test_split_ratio', type=float, default=0.3)
@click.option('--batch_size', type=int, default='64')
@click.option('--img_resize_size', type=int, default='64')
@click.option('--lr', type=float, default=0.0001)
@click.option('--num_epochs', type=int, default=5)
@click.option('--shuffle', type=bool, default=False)
@click.option('--load_model', type=str, default=None)
@click.option('--sample_submit_file_path', type=click.Path(), default = '../datas/sample_submission.csv')
def main(mode, exp_path, train_csv_path, test_csv_path, project_name, wandb_logging, wandb_account_entity, run_name, seed, test_split_ratio, batch_size,
         img_resize_size, lr, num_epochs, shuffle, load_model, sample_submit_file_path, keep_train, keep_train_model_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config(locals().copy())
    seed_everything(config.settings['seed'])


    if mode=="train":
        log_path = exp_path + run_name + '/'
        log_path = os.path.normpath(log_path) + '/'
        
        if not os.path.exists(log_path):
            assert keep_train is True, 'There is no trained model to resume!'
            os.makedirs(log_path)

        if wandb_logging is True:
            wandb = wandb_login()
            wandb.init(project=config.settings['project_name'], 
                       entity=config.settings['wandb_account_entity'], 
                       reinit=True, 
                       name=config.settings['run_name'])
            update_wandb_config(wandb, config)
        else:
            wandb = None
            config.save_config(log_path+'config.json')

        label_encoder, train_df, val_df = label_preprocessing(config.settings['train_csv_path'], 
                                                              config.settings['test_split_ratio'])
        
        if keep_train:
            with open(log_path+'label_encoder.pkl', 'rb') as file:
                label_encoder = pickle.load(file)
        else:
            with open(log_path + 'label_encoder.pkl', 'wb') as file:
                pickle.dump(label_encoder, file)

        train_loader = get_train_loader(train_df, 
                                        config.settings['img_resize_size'],
                                        config.settings['batch_size'],
                                        config.settings['shuffle'])
        val_loader = get_val_loader(val_df,
                                    config.settings['img_resize_size'],
                                    config.settings['batch_size'],
                                    config.settings['shuffle'])
    
    
        model = BaseModel(label_encoder)
        model.to(device)
        print(f"Model is on {next(model.parameters()).device}")
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=config.settings['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8)
        loss_func = torch.nn.CrossEntropyLoss()

        Trainer = BaseTrainer(model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            loss_func=loss_func,
                            num_epochs=config.settings['num_epochs'],
                            device=device,
                            save_path= log_path,
                            scheduler=scheduler,
                            wandb=wandb)

        keep_train_model_path = log_path + keep_train_model_file
        best_model = Trainer.train(keep_train=keep_train,
                                   keep_train_model_file=keep_train_model_path)


    elif mode=="inference":
        log_path = exp_path + run_name + '/'
        log_path = os.path.normpath(log_path) + '/'

        with open(log_path+'label_encoder.pkl', 'rb') as file:
            label_encoder = pickle.load(file)

        test_loader = get_test_loader(config.settings['test_csv_path'],
                                      config.settings['img_resize_size'],
                                      config.settings['batch_size'],
                                      config.settings['shuffle']
                                      )
        if load_model is None:
            ValueError("Model Path Error!")
        model = torch.jit.load(log_path + config.settings['load_model'],
                               map_location=torch.device(device))
        
        preds = inference(model, test_loader, label_encoder, device)
        
        if os.path.exists(sample_submit_file_path):
            submit_save_path = log_path + f'submit_{run_name}.csv'
            make_submit(preds, sample_submit_file_path, submit_save_path)
        else:
            print("Sample submit files is not exists!")

if __name__ =="__main__":
    main()