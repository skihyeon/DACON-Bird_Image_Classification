from torch.utils.data import Dataset
import cv2
import os

class BirdDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
    
    def __getitem__(self, index):
        try:
            img_path = self.img_path_list[index]
            img_path = img_path.replace("./", "")
            img_path = '../datas/' + img_path
            if self.is_running_in_colab():
                img_path = os.getcwd() + '/' + img_path
                img_path = os.path.normpath(img_path)
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

            img = self.transforms(image=img)['image'] if self.transforms is not None else img

            label = self.label_list[index] if self.label_list is not None else None
            return img, label
        except Exception as e:
            print(f"인덱스 {index}에서 오류 발생: {str(e)}")
            print(f"문제의 이미지 경로: {img_path}")
            raise
        
    def __len__(self):
        return len(self.img_path_list)
    
    def is_running_in_colab(self):
        try:
            import google.colab
            return True
        except ImportError:
            return False