# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import svm
import os
import argparse

'''
将用dlib提取的特征和用model提取的特征合并作为所有的特征，利用svm网络训练分类测试
'''

def train(args):
    classes = 3

    print('loading data...')
    f_data = np.load(args.feature_dlib_train)
    f_vgg_data = np.load(args.feature_model_train)
    train_data = np.concatenate([f_vgg_data, f_data])
    # train_data = np.expand_dims(train_data, axis=-1)

    train_csv = pd.read_csv(args.train_label_csv)
    train_label = [i for i in train_csv['lianxing']]
    print('train data shape: ', train_data.shape, 'label shape: ', np.shape(train_label))

    if args.feature_dlib_val:
        f_data = np.load(args.feature_dlib_val)
        f_vgg_data = np.load(args.feature_model_val)
        val_data = np.concatenate([f_vgg_data, f_data])
        # val_data = np.expand_dims(val_data, axis=-1)

        val_csv = pd.read_csv(args.val_label_csv)
        val_label = [i for i in val_csv['lianxing']]
        print('val data shape: ', val_data.shape, 'label shape: ', np.shape(val_label))
    print('Data is done!')

    # train
    print('Trainning model......')
    clf = svm.SVC()
    clf.fit(train_data, train_label)
    if args.feature_dlib_val:
        score = clf.score(val_data, val_label)
        print('val score:', score)

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

    args = parse.parse_args()
    train(args)
