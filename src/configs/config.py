
class Config():
    class general:
        train_csv_path = "../datas/train.csv"
        test_csv_path = '../datas/test.csv'
        test_split_ratio = 0.3
        resize_img_size = 64
        
    class train:
        lr = 0.001
        batch_size = 64
        shuffle = False
        num_epochs = 1
        save_path = '../model_files/model.pt'
    
    class inference:
        model_path = '../model_files/model.pt'
        batch_size = 64
        shuffle = False
    pass