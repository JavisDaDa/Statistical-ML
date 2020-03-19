import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger
import datautils
from keras.layers import Input
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
from keras import backend as K
from keras.utils import multi_gpu_model
import tensorflow as tf
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class ResNet(object):
    def __init__(self, train_df, val_df, train_dir, val_dir, class_dic, test_df, test_dir, train_sample=50000,
                 val_sample=10000, image_size=256, batch_size=32, num_classes=251, epoch=10):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.test_df['label'] = self.test_df['label'].astype(str)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.class_dic = class_dic
        self.train_sample = train_sample
        self.val_sample = val_sample
        self.test_sample = len(test_df)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epoch = epoch
        self.model_name = f'ResNet_{image_size}_{train_sample}.hdf5'
        self.model = None
        self.history = None
        self.train_generator = None

    def train(self, optimizer='adam', loss='categorical_crossentropy', metrics='accuracy'):
        train_generator = datautils.getDeepTrain(self.train_df, self.train_dir, sample=self.train_sample,
                                                 image_size=self.image_size, batch_size=self.batch_size)
        val_generator = datautils.getDeepTest(self.val_df, self.val_dir, sample=self.val_sample,
                                              image_size=self.image_size, batch_size=self.batch_size)
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = val_generator.n // val_generator.batch_size

        self.model.compile(optimizer, loss=loss, metrics=[metrics])
        checkpointer_path = f'ResNet_{self.image_size}_{self.train_sample}_checkpoint.hdf5'
        csv_logger_path = f'history_{self.num_classes}.log'
        checkpointer = ModelCheckpoint(filepath=checkpointer_path, verbose=1, save_best_only=True)
        csv_logger = CSVLogger(csv_logger_path)
        history = self.model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      validation_data=val_generator,
                                      validation_steps=STEP_SIZE_VALID,
                                      epochs=self.epoch,
                                      verbose=1,
                                      callbacks=[csv_logger, checkpointer]
                                      )
        self.model.save(self.model_name)
        self.history = history
        self.train_generator = train_generator

    def build_model(self):
        SHAPE = (self.image_size, self.image_size, 3)
        input_ = Input(shape=SHAPE)
        conv1 = self.conv2d_bn(input_, 64, kernel_size=(7, 7), strides=(2, 2))
        pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

        conv2 = self.residual_block(64, 2, is_first_layer=True)(pool1)
        conv3 = self.residual_block(128, 2, is_first_layer=True)(conv2)
        conv4 = self.residual_block(256, 2, is_first_layer=True)(conv3)
        conv5 = self.residual_block(512, 2, is_first_layer=True)(conv4)

        pool2 = GlobalAvgPool2D()(conv5)
        output_ = Dense(self.num_classes, activation='softmax')(pool2)

        with tf.device('/cpu:0'):
            model = Model(inputs=input_, outputs=output_)
            model.summary()
        self.model = model

    def conv2d_bn(self, x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
        """
        conv2d -> batch normalization -> relu activation
        """
        x = Conv2D(nb_filter, kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   kernel_regularizer=regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def shortcut(self, input, residual):
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_height = int(round(input_shape[1] / residual_shape[1]))
        stride_width = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]

        identity = input
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            identity = Conv2D(filters=residual_shape[3],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_regularizer=regularizers.l2(0.0001))(input)

        return add([identity, residual])

    def basic_block(self, nb_filter, strides=(1, 1)):
        def f(input):
            conv1 = self.conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
            residual = self.conv2d_bn(conv1, nb_filter, kernel_size=(3, 3))

            return self.shortcut(input, residual)
        return f

    def residual_block(self, nb_filter, repetitions, is_first_layer=False):
        def f(input):
            for i in range(repetitions):
                strides = (1, 1)
                if i == 0 and not is_first_layer:
                    strides = (2, 2)
                input = self.basic_block(nb_filter, strides)(input)
            return input
        return f

    def plot_loss(self):
        plt.plot(self.history['loss'])
        plt.title('Loss')
        plt.show()

    def evaluate(self, save=False):
        if save:
            model = load_model(self.model_name)
        else:
            model = self.model
        val_datagen = datautils.getDeepTest(self.val_df, self.val_dir, sample=self.val_sample,
                                            image_size=self.image_size, batch_size=self.batch_size, class_mode='class')
        val_loss, val_acc = model.evaluate_generator(val_datagen)
        print(f'validation loss is {val_loss}, validation accuracy is {val_acc}')
        return val_loss, val_acc

    def test(self, save=False):
        if save:
            model = load_model(self.model_name)
        else:
            model = self.model
        test_datagen = datautils.getDeepTest(self.test_df, self.test_dir, sample=self.test_sample,
                                             image_size=self.image_size, batch_size=self.batch_size, class_mode='label')
        preds = model.predict_generator(test_datagen)
        trueclass = datautils.preds2trueclass(preds, test_datagen, train_datagen=self.train_generator, test=True)
        truelabel = datautils.trueclass2label(trueclass, self.class_dic)
        df, combinedf = datautils.save2csv(test_datagen, truelabel, test_df=self.test_df, save=True,
                                           name=f'submission_{self.model_name}')
        return df, combinedf