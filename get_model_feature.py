from keras.models import load_model
from keras import backend as K
import pandas as pd
import numpy as np
import pickle

'''
根据训练的脸型模型，提取倒数第二层全连接层的信息，作为特征
'''

if __name__ == '__main__':
    model = load_model('./model/vggface1-weights-improvement-03-0.82.hdf5')
    layer_1 = K.function([model.layers[0].input], [model.layers[-2].output])

    # data
    print('read data...')
    train_data = np.load('./train/images.npy')
    train_data = train_data / 255.0
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    print('train_data shape: ', train_data.shape)
    print('Data is done!')

    batch = 32
    print('get features...')
    for i in range(int(np.ceil(len(train_data) / batch))):
        if (i + 1) * batch > len(train_data):
            batch_data = train_data[i * batch: len(train_data)]
        else:
            batch_data = train_data[i * batch: (i + 1) * batch]
        if i == 0:
            f_vgg = layer_1([batch_data])[0]
            # print(np.shape(f_vgg))
        else:
            f = layer_1([batch_data])[0]
            f_vgg = np.concatenate([f_vgg, f])

        if i % 100 == 0:
            print(i)
    print(f_vgg.shape)
    np.save('./train/features_model_train.npy', f_vgg)
    print('Done...')