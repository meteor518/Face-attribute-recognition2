# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from FaceDect import *
import numpy as np
import cv2

'''
根据dlib的68点，提取脸型轮廓信息，如轮廓上所有点的：两点的斜率、三角夹角、轮廓曲率等
'''


# 计算三个点两向量之间的夹角cos
def compute_cos(a, b, c):
    mouth21 = a - b
    mouth23 = c - b
    return mouth21.dot(mouth23) / np.sqrt(mouth21.dot(mouth21)) / np.sqrt(mouth23.dot(mouth23))


# 计算曲率
def compute_curvature(a):
    dx_dt = np.gradient(a[:, 0])
    dy_dt = np.gradient(a[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    count = int(len(a) / 2)
    return curvature[count]


datas = np.load('./result/xy68_val.npy')
features = []

if __name__ == '__main__':
    # # 将dlib打点画图显示
    # print('plot dlib 68 dot')
    # train_data = np.load('./train/images.npy')
    # features = []
    # for i in range(4):
    #     img = train_data[i]
    #     print(img.shape)
    #     emptyImage = np.zeros(img.shape, np.uint8)
    #
    #     face_landmark = np.zeros((68, 2))
    #     dets = detector(img, 1)
    #     for k, d in enumerate(dets):
    #         shape = predictor(img, d)
    #         for i in range(68):
    #             face_landmark[i, 0] = shape.part(i).x
    #             face_landmark[i, 1] = shape.part(i).y
    #             cv2.circle(img, (int(face_landmark[i, 0]), int(face_landmark[i, 1])), 1, (0, 0, 255))
    #             cv2.circle(emptyImage, (int(face_landmark[i, 0]), int(face_landmark[i, 1])), 1, (255, 0, 0))
    #     fig = plt.figure(figsize=(6 * 2, 6))
    #     plt.subplot(121)
    #     plt.imshow(img)
    #     plt.subplot(122)
    #     plt.imshow(emptyImage)
    #     plt.show()

    # 提取68点的坐标

    print('get images 68 dots information...')
    train_data = np.load('./train/images.npy')
    data_68 = []
    for i in range(len(train_data)):
        img = train_data[i]
        temp = feature_68(img)
        data_68.append(temp)
        if i % 1000 == 0:
            print('Having done {} images'.format(i))

    print(np.shape(data_68))
    np.save('./train/xy68_train.npy', data_68)

    # 根据68点坐标信息，计算轮廓的特征，如两点斜率，三点间夹角，轮廓曲率等等
    print('get face feature...')
    features = []
    for i in range(len(data_68)):
        data17 = data_68[i][:17]
        data_ = []
        # cos角
        for j in range(0, 15):
            data_.append(compute_cos(data17[j], data17[j + 1], data17[j + 2]))
        # 斜率
        for j in range(1, 15):
            slope = data17[j, :] - data17[j + 1, :]
            data_.append(slope[1] / slope[0])
        # 曲率
        for j in range(0, 15):
            data_.append(compute_curvature(data17[j:j + 3, :]))

        features.append(data_)

    print(np.shape(features))
    np.save('./train/features_dlib68_train.npy', features)
    print('Done...')
