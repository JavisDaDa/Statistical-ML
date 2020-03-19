import shutil
import traceback
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array
from keras_preprocessing.image import ImageDataGenerator

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
    train = []
    for i in range(n):
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

def gettrain(train_df, train_dir, sample=50000, image_size=32, batch_size=1):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_df = train_df.sample(sample)
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='img_name',
        y_col='class',
        class_mode='categorical',
        directory=train_dir,  # this is the target directory
        shuffle=False,
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels
    y_train = np.array(train_generator.classes)
    train_generator.reset()
    X_train = []
    for i in range(sample):
        img, _ = next(train_generator)
        img = img[0].reshape((-1,))
        X_train.append(img)
    X_train = np.array(X_train)
    return X_train, y_train

def gettestdata(df,dir, sample=50000, image_size=32, batch_size=1, class_mode='class'):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_df = df.sample(sample)
    train_generator = test_datagen.flow_from_dataframe(
        train_df,
        x_col='img_name',
        y_col=class_mode,
        class_mode='categorical',
        directory=dir,  # this is the target directory
        shuffle=False,
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels
    y = np.array(train_generator.classes)
    train_generator.reset()
    X = []
    for i in range(sample):
        img, _ = next(train_generator)
        img = img[0].reshape((-1,))
        X.append(img)
    X = np.array(X)
    labels = (train_generator.class_indices)
    label = dict((v, k) for k, v in labels.items())
    return X, y, label


def getDeepTrain(train_df, train_dir, sample=50000, image_size=32, batch_size=1):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_df = train_df.sample(sample)
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='img_name',
        y_col='class',
        class_mode='categorical',
        directory=train_dir,  # this is the target directory
        shuffle=False,
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels
    return train_generator

def getDeepTest(df,dir, sample=50000, image_size=32, batch_size=1, class_mode='class'):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_df = df.sample(sample)
    train_generator = test_datagen.flow_from_dataframe(
        train_df,
        x_col='img_name',
        y_col=class_mode,
        class_mode='categorical',
        directory=dir,  # this is the target directory
        shuffle=False,
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=batch_size)
    return train_generator


def preds2trueclass(preds, datagen, train_datagen=None, label=None, test=False):
    p = np.argmax(preds, axis=1)
    if not test:
        labels = datagen.class_indices
        label = dict((v, k) for k, v in labels.items())
    else:
        labels = train_datagen.class_indices

        label = dict((v, k) for k, v in labels.items())
    trueclass = []
    for i in list(p):
        trueclass.append(label[i])
    return trueclass

def createclass2label(path):
    table = pd.read_table(path, header=None, sep=' ')
    table.columns = ['label', 'class']
    class_dic = {}
    n = len(table)
    for i in range(n):
        class_dic[table.iloc[i]['class']] = table.iloc[i]['label']
    return class_dic

def trueclass2label(trueclass, class_dir):
    truelabel = []
    for i in trueclass:
        truelabel.append(class_dir[i])
    return truelabel

def save2csv(datagen, truelabel, test_df=None, save=False, name='submission'):
    files = datagen.filenames
    pred_dfdata = {files[i]: truelabel[i] for i in range(len(truelabel))}
    df = pd.DataFrame([pred_dfdata]).T
    df = df.reset_index()
    df.columns = ['img_name', 'pred_label']
    combinedf = pd.merge(test_df, df)
    if save:
        combinedf.to_csv(f'{name}.csv')
    return df, combinedf