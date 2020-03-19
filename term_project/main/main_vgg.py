import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchvision import transforms, models
from tools.load_dataset import FoodDataset
from torch.utils.data import DataLoader


MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.001
log_interval = 10
val_interval = 1
N_CLASSES = 251
TRAIN_SAMPLE = 50000
VALID_SAMPLE = 5000
checkpoint_interval = 2

train_label_dir = os.path.join("..", "data", "train_labels.csv")
valid_label_dir = os.path.join("..", "data", "val_labels.csv")
test_label_dir = os.path.join("..", "data", "sample_submission.csv")

train_df = pd.read_csv(train_label_dir)
valid_df = pd.read_csv(valid_label_dir)

train_data_df = train_df.sample(TRAIN_SAMPLE)
valid_data_df = valid_df.sample(VALID_SAMPLE)
test_data_df = pd.read_csv(test_label_dir)


# data preparation
data_dir = os.path.join("D:\Rice\COMP 540", "data")
train_dir = os.path.join(data_dir, "train_set")
valid_dir = os.path.join(data_dir, "val_set")
test_dir = os.path.join(data_dir, "test_set")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

train_data = FoodDataset(data_dir=train_dir, data_df=train_data_df, transform=train_transform)
valid_data = FoodDataset(data_dir=valid_dir, data_df=valid_data_df, transform=valid_transform)
test_data = FoodDataset(data_dir=test_dir, data_df=test_data_df, transform=test_transform)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

# choose model
vgg_model = models.vgg16(pretrained=True)
for parameter in vgg_model.parameters():
    parameter.requires_grad = False

classifier = nn.Sequential(OrderedDict({
    'fc1': nn.Linear(25088, 5000),
    'relu': nn.ReLU(),
    'drop': nn.Dropout(p=0.5),
    'fc2': nn.Linear(5000, N_CLASSES),
    'output': nn.LogSoftmax(dim=1)
}))

vgg_model.classifier = classifier




# def validation(model, validateloader, criterion):
#     val_loss = 0
#     accuracy = 0
#
#     for images, labels in iter(validateloader):
#         images, labels = images.to('cuda'), labels.to('cuda')
#
#         output = model.forward(images)
#         val_loss += criterion(output, labels).item()
#
#         probabilities = torch.exp(output)
#
#         equality = (labels.data == probabilities.max(dim=1)[1])
#         accuracy += equality.type(torch.FloatTensor).mean()
#
#     return val_loss, accuracy

# Loss Function
criterion = nn.NLLLoss()
# optimizer
optimizer = optim.Adam(vgg_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
# train

def train_classifier(model, print_every):
    train_curve = list()
    valid_curve = list()
    model.to('cuda')
    start_epoch = -1
    for e in range(start_epoch + 1, MAX_EPOCH):
        loss_mean = 0.
        correct = 0.
        total = 0.

        model.train()
        for i, data in enumerate(train_loader):

            # forward
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            output = model.forward(images)

            # backward
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # calculate loss and training infomation
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().numpy()

            # print
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % print_every == 0:
                loss_mean = loss_mean / print_every
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    e, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step()

        if (e + 1) % checkpoint_interval == 0:

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": e
            }
            path_checkpoint = f'.checkpoint_{e}_epoch.pkl'
            torch.save(checkpoint, path_checkpoint)

        if (e + 1) % val_interval == 0:
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().numpy()

                    loss_val += loss.item()

                valid_curve.append(loss.item())
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    e, MAX_EPOCH, j + 1, len(valid_loader), loss_val / len(valid_loader), correct_val / total_val))


train_classifier()
def save_model(model, name, save_state_dic=False):
    model_path = f'./{name}.pkl'
    if save_state_dic:
        path_state_dict = f'./{name}_state_dict.pkl'
        model_state_dict = model.state_dict()
        torch.save(model_state_dict, path_state_dict)
    torch.save(model, model_path)


def load_model(model_path, state_dic_path=None):

    model_load = torch.load(model_path)
    if state_dic_path is not None:
        state_dict_load = torch.load(state_dic_path)
        return model_load, state_dict_load
    return model_load


def train_classifier_resume(model, optimizer, path_checkpoint):
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    train_curve = list()
    valid_curve = list()

    print_every = 40

    vgg_model.to('cuda')
    for e in range(start_epoch + 1, MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        model.train()
        for i, data in enumerate(train_loader):

            # forward
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            output = model.forward(images)

            # backward
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # calculate loss and training infomation
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().numpy()

            # print
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % print_every == 0:
                loss_mean = loss_mean / print_every
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    e, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        scheduler.step()

        if (e + 1) % checkpoint_interval == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": e
            }
            path_checkpoint = f'.checkpoint_{e}_epoch.pkl'
            torch.save(checkpoint, path_checkpoint)

        if (e + 1) % val_interval == 0:
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().numpy()

                    loss_val += loss.item()

                valid_curve.append(loss.item())
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    e, MAX_EPOCH, j + 1, len(valid_loader), loss_val / len(valid_loader), correct_val / total_val))

# def test_accuracy(model, test_loader):
#     model.eval()
#     model.to('cuda')
#     with torch.no_grad():
#         for images, _ in test_loader:
#             images = images.to('cuda')
#             output = model.forward(images)
#             probabilities = torch.exp(output)
#             labels = probabilities.max(dim=1)[1]
