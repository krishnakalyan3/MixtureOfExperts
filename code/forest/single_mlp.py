#!/usr/bin/env python3

import sys

sys.path.append('../helper_utils')
sys.path.append('/home/kkalyan3/code/helper_utils')
import logging
import time
from keras.layers import Dense, Activation
from keras.models import Sequential
from utils import load_array, max_model, max_transform
from keras import optimizers

HELIOS = '/home/kkalyan3/code/forest/CONFIG_EXP100.json'

class MLP(object):
    def __init__(self):
        pass


    def mlp_model_mse(self):
        model = Sequential()
        model.add(Dense(150, input_dim=54))
        model.add(Activation('tanh'))
        model.add(Dense(50))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('tanh'))
        sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
        return model


if __name__ == '__main__':
    logging.basicConfig(filename='../log/forest_single_mlp.txt', level=logging.INFO)
    start_time = time.time()
    logging.info('##### NEW EXPERIMENT #####')

    TRAIN = '/home/kkalyan3/data/train/100/'
    TEST = '/home/kkalyan3/data/test/100/'
    VAL = '/home/kkalyan3/data/val/100/'

    x_train = load_array(TRAIN + 'X_train.bc/')
    y_train = load_array(TRAIN + 'y_train.bc/')

    x_test = load_array(TEST + 'X_test.bc/')
    y_test = load_array(TEST + 'y_test.bc/')

    x_val = load_array(VAL + 'X_val.bc/')
    y_val = load_array(VAL + 'y_val.bc/')

    logging.info('Training Size ' + str(x_train.shape[0]))
    logging.info('Testing Size ' + str(x_test.shape[0]))

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    y_val[y_val == 0] = -1

    max_values = max_model(x_train)
    x_train = max_transform(max_values, x_train)
    x_test = max_transform(max_values, x_test)
    x_val = max_transform(max_values, x_val)
    logging.info('Scaling Finished')

    mlp = MLP()
    model = mlp.mlp_model_mse()
    model.fit(x_train, y_train, batch_size=256, verbose=1,
              validation_data=(x_val, y_val), epochs=500)

    yhats_train = model.evaluate(x_train, y_train)
    yhats_val = model.evaluate(x_val, y_val)
    yhats_test = model.evaluate(x_test, y_test)

    tr_e = (1 - yhats_train[1]) * 100
    tt_e = (1 - yhats_test[1]) * 100
    val_e = (1 - yhats_val[1]) * 100

    total_time = (time.time() - start_time) / 60

    logging.info('Train Error, Validation Error, Test Error, Time Taken')
    logging.info('{}, {}, {}, {}'.format(tr_e, val_e, tt_e, total_time))

    logging.info('##### EXPERIMENT COMPLETE #####')
