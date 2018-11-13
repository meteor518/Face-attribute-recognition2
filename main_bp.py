# -*- coding: utf-8 -*-
from keras.utils import to_categorical
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.models import *
import pandas as pd
import numpy as np

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import argparse

'''
将用dlib提取的特征和用model提取的特征合并作为所有的特征，利用BP网络再训练
'''

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 进行配置，使用70%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)


def Net(input_len=15, classes=3):
    inputs = Input((input_len,))
    x = inputs

    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs, x)
    return model


def train(args):
    classes = 3

    print('loading data...')
    f_data = np.load(args.feature_dlib_train)
    f_vgg_data = np.load(args.feature_model_train)
    train_data = np.concatenate([f_vgg_data, f_data])
    # train_data = np.expand_dims(train_data, axis=-1)

    train_csv = pd.read_csv(args.train_label_csv)
    train_label = [i for i in train_csv['lianxing']]
    train_label = to_categorical(train_label, num_classes=classes)
    print('train data shape: ', train_data.shape, 'label shape: ', np.shape(train_label))

    if args.feature_dlib_val:
        f_data = np.load(args.feature_dlib_val)
        f_vgg_data = np.load(args.feature_model_val)
        val_data = np.concatenate([f_vgg_data, f_data])
        # val_data = np.expand_dims(val_data, axis=-1)

        val_csv = pd.read_csv(args.val_label_csv)
        val_label = [i for i in val_csv['lianxing']]
        val_label = to_categorical(val_label, num_classes=classes)
        print('val data shape: ', val_data.shape, 'label shape: ', np.shape(val_label))
    print('Data is done!')

    # model
    print('get model....')
    model_name = 'feature_lianxiang'
    model = Net(len(train_label[0]), classes)

    model.summary()
    opt = Adam(lr=1e-3, decay=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # train
    print('Trainning model......')
    tensorboard_log = TensorBoard('./feature_tensorboard/' + model_name, write_graph=True, histogram_freq=0)
    csv = CSVLogger(model_name + '.csv')

    if args.feature_dlib_val:
        # callbacks
        model_checkpoint = ModelCheckpoint(model_name + '.h5', monitor='val_acc', verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor='val_acc', patience=15, verbose=2)

        model.fit(train_data, train_label, batch_size=64, epochs=1000,
                  callbacks=[model_checkpoint, csv, tensorboard_log, early_stop],
                  verbose=1, shuffle=True, validation_data=(val_data, val_label))
    else:
        # callbacks
        model_checkpoint = ModelCheckpoint(model_name + '.h5', monitor='acc', verbose=1, save_best_only=True)
        model.fit(train_data, train_label, batch_size=args.batch_size, epochs=args.epochs,
                  callbacks=[model_checkpoint, csv, tensorboard_log],
                  verbose=1, shuffle=True)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--feature-dlib-train', '-fdt', required=True,
                       help='the .npy file of features based dilb for training')
    parse.add_argument('--feature-model-train', '-fmt', required=True,
                       help='the .npy file of features based model for training')
    parse.add_argument('--train-label-csv', '-tl', required=True, help='.csv file for training')

    parse.add_argument('--feature-dlib-val', '-fdv', help='the .npy file of features based dilb for validation')
    parse.add_argument('--feature-model-val', '-fmv', help='the .npy file of features based model for validation')
    parse.add_argument('--val-label-csv', '-vl',help='.csv file for validation')

    parse.add_argument('--batch-size', '-batch', type=int, default=128)
    parse.add_argument('--epochs', '-e', type=int, default=1000)

    args = parse.parse_args()
    train(args)
