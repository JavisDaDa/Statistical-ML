import os
import random
from PIL import Image
from torch.utils.data import Dataset
from datautils import createclass2label

random.seed(1)
path = os.path.join("..", "data", "class_list.txt")
# food_label = createclass2label(path)

class FoodDataset(Dataset):
    def __init__(self, data_dir, data_df, transform=None, class_list_dir=path):
        self.food_label = createclass2label(class_list_dir)
        self.data_dir = data_dir
        self.data_info = data_df
        self.transform = transform

    def __getitem__(self, index):
        # path_img = os.path.join(self.data_dir, self.data_info.loc[index, 'img_name'])
        # label = self.data_info.loc[index, 'label']
        path_img = os.path.join(self.data_dir, self.data_info.iloc[index]['img_name'])
        label = self.data_info.iloc[index]['label']
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)
