import os
import platform
import numpy as np
from keras.datasets.cifar import load_batch
from tensorflow.keras.datasets import cifar10
from keras import backend as K


def load_cifar10():
    if platform.system() != "Darwin":
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()
        return (train_x, train_y), (test_x, test_y)

    dpath = os.environ["DATASETS"]

    if (dpath == None):
        print("Missing DATASETS env var.")
        exit(-1)

    path = os.path.join(dpath, "cifar10")

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)