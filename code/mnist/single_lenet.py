#!/usr/bin/env python3

# This should give us 5 % error
import sys
sys.path.append('../helper_utils')
sys.path.append('/home/kkalyan3/code/helper_utils')
from helper_callbacks import CustomCallback
import time
from nn_arch import nn_models
from keras.utils import np_utils
import numpy as np
import logging
from utils import load_array, eval_target

PATH = '/home/kkalyan3/data/mnist/'
#PATH = '/gel/usr/skkal1/SVM-Experts/data/mnist/'
#PATH = '../../data/mnist/'

if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(filename='../log/mnist_lent' + str(start_time), level=logging.INFO)

    logging.info('##### NEW EXPERIMENT_' + str(start_time) + '_#####')

    TRAIN = PATH + 'train/500/'
    TEST = PATH + 'test/500/'
    VAL = PATH + 'val/500/'

    x_train = load_array(TRAIN + 'x_train.bc/')
    y_train = load_array(TRAIN + 'y_train.bc/')

    x_test = load_array(TEST + 'x_test.bc/')
    y_test = load_array(TEST + 'y_test.bc/')

    x_val = load_array(VAL + 'x_val.bc/')
    y_val = load_array(VAL + 'y_val.bc/')

    logging.info('Training Size ' + str(x_train.shape[0]))
    logging.info('Testing Size ' + str(x_test.shape[0]))
    logging.info('Val Size ' + str(x_val.shape[0]))

    img_rows, img_cols = 28, 28
    num_pixels = img_rows * img_cols

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

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

    model = nn_models()
    model.ip_shape = x_train.shape
    model = model.lenet5()
    early_callback = CustomCallback()

    model.fit(x_train, y_train, batch_size=256, verbose=1,
              validation_data=(x_val, y_val), epochs=500, callbacks=[early_callback])

    yhats_train = model.predict(x_train, batch_size=256)
    yhats_val = model.predict(x_val, batch_size=256)
    yhats_test = model.predict(x_test, batch_size=256)

    train_error = eval_target(yhats_train, y_train)
    val_error = eval_target(yhats_val, y_val)
    test_error = eval_target(yhats_test, y_test)

    total_time = (time.time() - start_time)/60
    print(total_time)

    logging.info('{}, {}, {}, {}'.format("Training Error", "Val Error", "Test Error", "Time"))
    logging.info('{}, {}, {}, {}'.format(train_error, val_error, test_error, total_time))
    logging.info('##### EXPERIMENT COMPLETE #####')
