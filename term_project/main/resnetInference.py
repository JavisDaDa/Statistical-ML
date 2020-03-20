import os
import pandas as pd
import torch
import time
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tools.load_dataset import FoodDataset
from torch.utils.data import DataLoader


N_CLASSES = 251
BATCH_SIZE = 32
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
test_label_dir = os.path.join("..", "data", "sample_submission.csv")
test_data_df = pd.read_csv(test_label_dir)

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

def img_transfoem(img_rgb, transform=None):
    if transform is None:
        raise ValueError('Please add transforms to images')
    img_t = transform(img_rgb)
    return img_t



data_dir = os.path.join("D:\Rice\COMP 540", "data")
test_dir = os.path.join(data_dir, "test_set")

# model path
model_path = "../model/resnet152/resnet152_state_dict.pkl"

time_total = 0
img_list, img_pred = list(), list()


# data
# img_names = list(test_data_df['img_name'])
# num_img = len(test_data_df)
test_data = FoodDataset(data_dir=test_dir, data_df=test_data_df, transform=test_transform)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
# model
resnet152 = models.resnet152()
num_features = resnet152.fc.in_features
resnet152.fc = nn.Linear(num_features, N_CLASSES)
state_dict = torch.load(model_path)
resnet152.load_state_dict(state_dict)
resnet152.to(device)
resnet152.eval()

pred_list = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        images, _ = data
        images = images.to(device)

        # path to img
        # img_rgb = Image.open(path_img).convert('RGB')

        # img to tensor
        # img_tensor = img_transfoem(img_rgb, test_transform)
        # img_tensor.unsqueeze_(0)
        # img_tensor = img_tensor.to(device)

        # tensor to vector
        time_tic = time.time()
        outputs = resnet152(images)
        time_toc = time.time()

        # convert to df
        _, pred_int = torch.max(outputs.data, 1)
        pred_list.append([i for i in pred_int.cpu().numpy().reshape((1, -1)).tolist()])
        print(f'Finished {i+1} prediction')
        if i > 3:
            break
print(pred_list)
# test_data_df['pred_label'] = pred_list
# test_data_df.to_csv('resnet152.csv')