#!/usr/bin/env python3

import sys

sys.path.append('/Users/krishna/MOOC/MixtureSVM/src/helper_utils')
sys.path.append('/home/kkalyan3/code/helper_utils')
from sklearn.neighbors import KNeighborsClassifier
import time
import numpy as np
from utils import load_array, eval_target
from keras.utils import np_utils
import logging


TRAIN_SIZE = 50000

PATH = '/home/kkalyan3/data/mnist/'
#PATH = '/gel/usr/skkal1/SVM-Experts/data/mnist/'
#PATH = '../../data/mnist/'

def knn_model(X, y):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    return neigh

if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(filename='../log/mnist_knn' + str(start_time) + '.txt', level=logging.INFO)

    logging.info('##### NEW EXPERIMENT_' + str(start_time) + '_#####')
    img_rows, img_cols = 28, 28
    num_pixels = img_rows * img_cols

    TRAIN = PATH + 'train/500/'
    TEST = PATH + 'test/500/'
    VAL = PATH + 'val/500/'

    x_train = load_array(TRAIN + 'x_train.bc/')
    y_train = load_array(TRAIN + 'y_train.bc/')
    x_test = load_array(TEST + 'x_test.bc/')
    y_test = load_array(TEST + 'y_test.bc/')
    x_val = load_array(VAL + 'x_val.bc/')
    y_val = load_array(VAL + 'y_val.bc/')

    x_train = x_train.reshape(x_train.shape[0], num_pixels)
    x_test = x_test.reshape(x_test.shape[0], num_pixels)
    x_val = x_val.reshape(x_val.shape[0], num_pixels)

    logging.info('Training Size ' + str(x_train.shape[0]))
    logging.info('Testing Size ' + str(x_test.shape[0]))
    logging.info('Val Size ' + str(x_val.shape[0]))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    y_val = np_utils.to_categorical(y_val, 10)

    x_train /= 255
    x_test /= 255
    x_val /= 255

    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    x_val -= mean_image

    model = knn_model(x_train, y_train)
    yhat_train = model.predict(x_train)
    yhat_test = model.predict(x_test)
    yhat_val = model.predict(x_val)

    train_error = eval_target(y_train, yhat_train)
    test_error = eval_target(y_test, yhat_test)
    val_error = eval_target(y_val, yhat_val)
    total_time = (time.time() - start_time)/60

    logging.info('{}, {}, {}, {}'.format("Training Error", "Val Error", "Test Error", "Time Taken"))
    logging.info('{}, {}, {}, {}'.format(train_error, val_error, test_error, total_time))

    print("Training Eval: ", train_error)
    print("Test Eval: ", test_error)
    print("Total TIme ", total_time)

    logging.info('##### EXPERIMENT COMPLETE #####')
