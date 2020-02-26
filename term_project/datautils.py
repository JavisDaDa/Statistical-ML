import shutil
import traceback
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array


def createTrainValiddf(train_dir, val_dir):
    train_df = pd.read_csv(train_dir)
    val_df = pd.read_csv(val_dir)
    train_df['path'] = train_df['img_name'].map(lambda x: os.path.join(train_dir, x))
    val_df['path'] = val_df['img_name'].map(lambda x: os.path.join(val_dir, x))
    train_df.set_index('label', drop=True, inplace=True)
    train_df.sort_index(ascending=True, inplace=True)
    val_df.set_index('label', drop=True, inplace=True)
    val_df.sort_index(ascending=True, inplace=True)
    return train_df, val_df


def moveTrainImages(train_df, train_dir):
    for i in range(251):
        fromimage = train_df.loc[i]['img_name'].values
        for img in fromimage:
            print(img)
            try:
                dis = os.path.join(f'../data/train_{i}')
                f_src = os.path.join(train_dir, img)
                if not os.path.exists(dis):
                    os.mkdir(dis)
                f_dis = os.path.join(dis, img)
                shutil.move(f_src, f_dis)
            except Exception as e:
                print('move_file ERROR')
                traceback.print_exc()


def moveValImages(val_df, val_dir):
    for i in range(251):
        fromimage = val_df.loc[i]['img_name'].values
        for img in fromimage:
            print(img)
            try:
                dis = os.path.join(f'../data/val_{i}')
                f_src = os.path.join(val_dir, img)
                if not os.path.exists(dis):
                    os.mkdir(dis)
                f_dis = os.path.join(dis, img)
                shutil.move(f_src, f_dis)
            except Exception as e:
                print('move_file ERROR')
                traceback.print_exc()

def load_data(df, dir, size=(256, 256, 3)):
    X = []
    y = []
    n = len(df)
    for i in range(n):
        train = []
        img_name = df.iloc[i]['img_name'].values
        label_name = df.iloc[i]['label'].values
        y.append(label_name)
        image = load_img(os.path.join(dir, img_name), color_mode='rgb', target_size=size)
        image = img_to_array(image)
        image = image.reshape((1, -1))
        train.append(np.squeeze(image))
    X.append(train)
    X = np.array(X)
    X = X.reshape((-1, X.shape[2]))
    y = np.array(y).reshape(-1, )
    return X, y


def load_batch(train_dir, val_dir, train_df, val_df, test_dir, test_df, size=(256, 256, 3)):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    train_length = len(train_df)
    val_length = len(val_df)
    test_length = len(test_df)
    for i in range(train_length):
        train = []
        img_name = train_df.iloc[i]['img_name'].values
        label_name = train_df.iloc[i]['label'].values
        y_train.append(label_name)
        image = load_img(os.path.join(train_dir, img_name), color_mode='rgb', target_size=size)
        image = img_to_array(image)
        image = image.reshape((1, -1))
        train.append(np.squeeze(image))
    X_train.append(train)
    X_train = np.array(X_train)
    X_train = X_train.reshape((-1, X_train.shape[2]))
    y_train = np.array(y_train).reshape(-1, )
    for i in range(val_length):
        train = []
        img_name = val_df.iloc[i]['img_name'].values
        label_name = val_df.iloc[i]['label'].values
        y_val.append(label_name)
        image = load_img(os.path.join(val_dir, img_name), color_mode='rgb', target_size=size)
        image = img_to_array(image)
        image = image.reshape((1, -1))
        train.append(np.squeeze(image))
    X_val.append(train)
    X_val = np.array(X_val)
    X_val = X_val.reshape((-1, X_val.shape[2]))
    y_val = np.array(y_val).reshape(-1, )
    for i in range(test_length):
        train = []


    return X_train, y_train, X_val, y_val