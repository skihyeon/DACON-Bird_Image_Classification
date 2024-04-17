from torch.utils.data import Dataset
import cv2
import os

class BirdDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_path = img_path.replace("./", "")
        img_path = '../datas/' + img_path
        if self.is_running_in_colab():
            print("Running in Colab!!!!")
            img_path = os.getcwd() + '/' + img_path
            img_path = os.path.normpath(img_path)
        img = cv2.imread(img_path)

        img = self.transforms(image=img)['image'] if self.transforms != None else img

        if self.label_list is not None:
            label = self.label_list[index]
            return img, label
        else:
            return img
    def __len__(self):
        return len(self.img_path_list)
    
    def is_running_in_colab(self):
        try:
            import google.colab
            return True
        except ImportError:
            return False