#!/usr/bin/env python3

import sys

sys.path.append('../helper_utils')
sys.path.append('/home/kkalyan3/code/helper_utils')
from nn_arch import nn_models
from keras.utils import np_utils
import numpy as np
import logging
import time
from helper_callbacks import CustomCallback
from utils import load_array, eval_target

INSTANCE = 1000

PATH = '/home/kkalyan3/data/mnist/'
#PATH = '/gel/usr/skkal1/SVM-Experts/data/mnist/'
#PATH = '../../data/mnist/'

class Uniform(object):
    def __init__(self):
        self.experts = None
        self.warn_log = [["Expert Training Error", "Expert Val Error"]]

    def model_train(self, X, y, x_val, y_val):
        model = nn_models()
        model.ip_shape = X.shape
        model = model.lenet5()
        early_callback = CustomCallback()
        model.fit(X, y, batch_size=256, epochs=500, validation_data=(x_val, y_val),
                  callbacks=[early_callback], verbose=1)

        yhat_train = model.predict(X, batch_size=256)
        yhat_val = model.predict(x_val, batch_size=256)

        train_error = eval_target(yhat_train, y)
        val_error = eval_target(yhat_val, y_val)
        self.warn_log.append([train_error, val_error])

        return model

    def main(self, x_train, y_train, x_test, y_test, x_val, y_val):
        yhats_train = 0
        yhats_test = 0
        yhats_val = 0

        for j in range(self.experts):
            model = self.model_train(x_train, y_train, x_val, y_val)
            yhats_train += model.predict(x_train, batch_size=256)
            yhats_test += model.predict(x_test, batch_size=256)
            yhats_val += model.predict(x_val, batch_size=256)

        yhats_train *= (1/self.experts)
        yhats_test *= (1/self.experts)
        yhats_val *= (1 / self.experts)

        train_error = eval_target(yhats_train, y_train)
        test_error = eval_target(yhats_test, y_test)
        val_error = eval_target(yhats_val, y_val)

        logging.info('{}, {}, {}'.format("Training Error", "Val Error", "Test Error"))
        logging.info('{}, {}, {}'.format(train_error, val_error, test_error))

        return None

if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(filename='../log/mnist_uniform'+ str(start_time), level=logging.INFO)

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

    uniform = Uniform()
    uniform.experts = 10
    uniform.main(x_train, y_train, x_test, y_test, x_val, y_val)

    total_time = (time.time() - start_time)/60
    logging.info("Total Time " + str(total_time))

    for i in uniform.warn_log:
        logging.warning(i)

    logging.info("#### Experiment Complete ####")
