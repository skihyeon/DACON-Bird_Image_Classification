from torch.utils.data import DataLoader
import pandas as pd
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import cv2

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from .mydataset import BirdDataset

def label_preprocessing(csv_file_path, test_split_ratio):
    df = pd.read_csv(csv_file_path)
    train, val, _, _ = train_test_split(df, df['label'], test_size=test_split_ratio, stratify=df['label'])

    le = preprocessing.LabelEncoder()
    train['label'] = le.fit_transform(train['label'])
    val['label'] = le.transform(val['label'])

    return le, train, val

def get_train_loader(encoded_train_df, resize_img_size, batch_size, shuffle=False):
    transform = albu.Compose([
                             albu.Resize(resize_img_size, resize_img_size, interpolation=cv2.INTER_LANCZOS4), 
                             albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                             albu.OneOf([
                                         albu.HorizontalFlip(p=1),
                                         albu.RandomRotate90(p=1),
                                         albu.VerticalFlip(p=1),
                                         ], p=1),
                            #  albu.OneOf([
                            #              albu.MotionBlur(p=1),
                            #              albu.OpticalDistortion(p=1),
                            #              albu.GaussNoise(p=1)
                            #              ], p=1),
                             ToTensorV2()
                            ])
    
    dataset = BirdDataset(encoded_train_df['img_path'].values, encoded_train_df['label'].values, transform)
    loader = DataLoader(dataset, batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    return loader

def get_val_loader(encoded_val_df, resize_img_size, batch_size, shuffle=False):
    transform = albu.Compose([
                             albu.Resize(resize_img_size, resize_img_size, interpolation=cv2.INTER_LANCZOS4),
                             albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                             ToTensorV2()
                            ])
    
    dataset = BirdDataset(encoded_val_df['img_path'].values, encoded_val_df['label'].values, transform)
    loader = DataLoader(dataset, batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    return loader


def get_test_loader(test_csv_path, resize_img_size, batch_size, shuffle=False):
    df = pd.read_csv(test_csv_path)
    transform = albu.Compose([
                             albu.Resize(resize_img_size, resize_img_size, interpolation=cv2.INTER_LANCZOS4),
                             albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
                             ToTensorV2()
                            ])
    
    dataset = BirdDataset(df['img_path'].values, None, transform)
    loader = DataLoader(dataset, batch_size, shuffle=shuffle,  pin_memory=True, num_workers=4)
    return loader