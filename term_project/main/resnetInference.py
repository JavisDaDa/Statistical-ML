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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
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
# inference

pred_list = []
state_dict = torch.load(model_path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
inference_model = models.resnet152()
num_features = inference_model.fc.in_features
inference_model.fc = nn.Linear(num_features, N_CLASSES)
inference_model.load_state_dict(state_dict)
inference_model.to(device)
inference_model.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        images, _ = data
        images = images.to(device)

        # tensor to vector
        outputs = inference_model(images)

        # convert to df
        _, pred_int = torch.max(outputs.data, 1)
        pred_list.append(pred_int.cpu().numpy().reshape((1, -1)).tolist())
        print(f'Finished {i+1} prediction')


new_pred_list = []
for i in range(len(pred_list)):
    for j in range(len(pred_list[i][0])):
        new_pred_list.append(pred_list[i][0][j])
test_data_df['pred_label'] = new_pred_list
test_data_df.to_csv('./drive/My Drive/COMP540/resnet152.csv')