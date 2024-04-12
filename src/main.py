import torch

from networks.basemodel import BaseModel
from utils.utils import seed_everything
from optim.trainer import BaseTrainer
from optim.inference import inference
from datasets.dataloader import label_preprocessing, get_train_loader, get_val_loader, get_test_loader
from configs.config import Config

def main(mode):
    seed_everything(456)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config()
    

    label_encoder, train_df, val_df = label_preprocessing(config.general.train_csv_path, 
                                                                  config.general.test_split_ratio)
    
    if mode=="train":
        train_loader = get_train_loader(train_df, 
                                        config.general.resize_img_size,
                                        config.train.batch_size,
                                        config.train.shuffle)
        val_loader = get_val_loader(val_df,
                                    config.general.resize_img_size,
                                    config.train.batch_size,
                                    config.train.shuffle)
    
    
        model = BaseModel(label_encoder)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.train.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8)
        loss_func = torch.nn.CrossEntropyLoss()

        Trainer = BaseTrainer(model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            loss_func=loss_func,
                            num_epochs=config.train.num_epochs,
                            device=device,
                            save_path=config.train.save_path,
                            scheduler=scheduler)

        best_model = Trainer.train()

    elif mode=="infer":
        test_loader = get_test_loader(config.general.test_csv_path,
                                      config.general.resize_img_size,
                                      config.inference.batch_size,
                                      config.inference.shuffle
                                      )
        
        model = torch.jit.load(config.inference.model_path,
                               map_location=torch.device(device))
        
        inference(model, test_loader, label_encoder, device)


if __name__ =="__main__":
    main("train")