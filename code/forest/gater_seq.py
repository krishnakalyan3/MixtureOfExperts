#!/usr/bin/env python3

# 50k test
# 10k val
# 100k train
# 50 experts
# 150 hidden units

# 500 units - 8.1 test error
# 150 units - 9.28 test error

import sys

sys.path.append('../helper_utils')
sys.path.append('/home/kkalyan3/code/helper_utils')
from keras.layers.merge import Dot
import json
from keras import optimizers
from keras.callbacks import TensorBoard
import time
import logging
from sklearn.svm import SVC
from keras.layers import Dense, Input
from keras.models import Model
import numpy as np
from utils import load_array, max_model, max_transform
from sklearn.utils import shuffle
import numpy
from helper_callbacks import CustomCallback
from sklearn.metrics import accuracy_score

numpy.set_printoptions(threshold=numpy.nan)


HELIOS = '/home/kkalyan3/code/forest/CONFIG_EXP100.json'
PATH = '/home/kkalyan3/data/forest/'

#HELIOS = 'CONFIG_EXP100.json'
#PATH = '../../data/forest/'


class MOE(object):
    def __init__(self):
        self.SEED = 1337
        self.exp = 'exp'
        self.iters = 3
        self.experts = 50
        self.gater_epoch = 500
        self.batch_size = 256
        self.hidden_units = 150
        self.learning_rate = 0.001
        self.bucket_dict = {}
        self.threshold_bucket = {}
        self.flag = 0
        self.random_search = False
        self.C = None
        self.gamma = None
        self.cache_size = 50000
        self.wm_xi = 0
        self.tf_log = None
        self.c = 1
        self.train_dim = None
        self.test_dim = None
        self.val_dim = None
        self.warn_log = [["Iter", "Expert Training Error", "Expert Val Error", "Expert Test Error"]]

        self.model = None

    def gater(self):
        dim_inputs_data = Input(shape=(self.train_dim[1],))
        dim_svm_yhat = Input(shape=(self.experts,))
        layer_1 = Dense(self.hidden_units,
                        activation='sigmoid')(dim_inputs_data)
        layer_2 = Dense(self.experts, name='layer_op_2',
                        activation='sigmoid', use_bias=False)(layer_1)
        layer_3 = Dot(1)([layer_2, dim_svm_yhat])
        out_layer = Dense(1, activation='tanh')(layer_3)
        model = Model(input=[dim_inputs_data, dim_svm_yhat], output=out_layer)
        adam = optimizers.Adam(lr=0.01)
        model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
        return model

    def svc(self, params):
        X, y = params
        X, y = shuffle(X, y, random_state=self.SEED)
        clf = SVC(C=self.C, kernel='rbf', gamma=self.gamma, cache_size=self.cache_size,
                  verbose=1, random_state=self.SEED)
        model = clf.fit(X, y)
        return model

    def moe_eval(self, gater_model, X, y, experts_out):
        yhat = gater_model.predict([X, experts_out], batch_size=self.batch_size)
        yhat[yhat >= 0] = 1
        yhat[yhat < 0] = -1
        acc = 100 * (np.sum(yhat.flatten() == y * 1.) / y.shape[0])
        return acc

    def get_expert(self, weight_data):
        thresh_hard = int(weight_data.shape[0] / weight_data.shape[1]) + self.c

        # [0.2, 0.1, 0.7, 0.1] -> [2, 0 ,1, 3]
        sort_index = np.argsort(-1 * weight_data)
        thresh_dict = {}

        # {0:0 , 1:0 , 2:0 ..}
        print(weight_data.shape)
        thresh_dict = thresh_dict.fromkeys(list(range(weight_data.shape[1])), 0)
        local_expert = {}

        for k, v in enumerate(sort_index):
            for i in v:
                if thresh_dict[i] < thresh_hard:
                    thresh_dict[i] += 1
                    if i not in local_expert:
                        local_expert[i] = [k]
                    else:
                        local_expert[i].append(k)
                    break
        return local_expert

    def get_random(self):
        local_expert = {}
        random_bucket = np.random.choice(self.experts, self.train_dim[0])
        for i, e in enumerate(random_bucket):
            if e not in local_expert:
                local_expert[e] = [i]
            else:
                local_expert[e].append(i)
        return local_expert

    def bucket_function(self, i):
        if i == 0:
            return self.get_random()
        else:
            return self.get_expert(self.wm_xi)

    def svc_model(self, X, y, x_test, y_test, x_val, y_val, i, j):
        X, y = shuffle(X, y, random_state=self.SEED)
        clf = SVC(C=self.C, kernel='rbf', gamma=self.gamma, cache_size=self.cache_size,
                  verbose=0, random_state=self.SEED)
        model = clf.fit(X, y)

        yhat_train = model.predict(X)
        yhat_val = model.predict(x_val)
        yhat_test = model.predict(x_test)

        train_error = (1 - accuracy_score(y, yhat_train)) * 100
        val_error = (1 - accuracy_score(y_val, yhat_val)) * 100
        test_error = (1 - accuracy_score(y_test, yhat_test)) * 100

        self.warn_log.append([i, train_error, val_error, test_error])

        return model


    @staticmethod
    def predict_models(X, model):
        return model.predict(X)

    @staticmethod
    def save_targets(split_buckets):
        sd = {}
        for k, v in split_buckets.items():
            for val in v:
                sd[val] = k
        expert_array = []
        for k1 in sorted(sd):
            expert_array.append(sd[k1])

        np.savetxt('temp/vis_final_10' + '.csv', expert_array, delimiter=",")
        return 0

    def train_model(self, X_train, y_train, X_test, y_test, X_val, y_val):

        for i in range(self.iters):
            split_buckets = self.bucket_function(i)
            experts_out_train = np.empty((self.train_dim[0], self.experts), dtype='float64')
            experts_out_test = np.empty((self.test_dim[0], self.experts), dtype='float64')
            experts_out_val = np.empty((self.val_dim[0], self.experts), dtype='float64')

            j = 0
            for expert_index in sorted(split_buckets):
                print("############################# Expert {} Iter {} ################################".format(j, i))
                X = X_train[split_buckets[expert_index]]
                y = y_train[split_buckets[expert_index]]
                model = self.svc_model(X, y, X_test, y_test, X_val, y_val, i, j)

                experts_out_train[:, expert_index] = model.predict(X_train)
                experts_out_test[:, expert_index] = model.predict(X_test)
                experts_out_val[:, expert_index] = model.predict(X_val)

                j += 1

            gater_model = self.gater()
            early_callback = CustomCallback()
            tb_callback = TensorBoard(log_dir=self.tf_log + str(i))
            history = gater_model.fit([X_train, experts_out_train], y_train, shuffle=True,
                                      batch_size=self.batch_size, verbose=1, validation_data=([X_val, experts_out_val], y_val),
                                      epochs=1000, callbacks=[tb_callback, early_callback])

            train_accuracy = self.moe_eval(gater_model, X_train, y_train, experts_out_train)
            test_accuracy = self.moe_eval(gater_model, X_test, y_test, experts_out_test)
            val_accuracy = self.moe_eval(gater_model, X_val, y_val, experts_out_val)

            print('Train Accuracy', train_accuracy)
            print('Test Accuracy', test_accuracy)
            print('Val Accuracy', val_accuracy)

            tre = 100 - train_accuracy
            tte = 100 - test_accuracy
            vale = 100 - val_accuracy
            expert_units = Model(inputs=gater_model.input,
                                outputs=gater_model.get_layer('layer_op_2').output)

            self.wm_xi = expert_units.predict([X_train, experts_out_train])

            logging.info('{}, {}, {}, {}'.format(i, tre, vale, tte))

        return None


if __name__ == '__main__':

    with open(HELIOS, 'r') as f:
        config = json.load(f)

    logging.basicConfig(filename='../log/forest_gater_seq_log.txt', level=logging.INFO)
    start_time = time.time()
    logging.info('##### NEW EXPERIMENT_' + str(start_time) + '_#####')

    logging.info(json.dumps(config, indent=4))
    TRAIN = PATH + 'train/100/'
    TEST = PATH + 'test/100/'
    VAL = PATH + 'val/100/'

    X_train = load_array(TRAIN + 'X_train.bc/')
    y_train = load_array(TRAIN + 'y_train.bc/')

    X_test = load_array(TEST + 'X_test.bc/')
    y_test = load_array(TEST + 'y_test.bc/')

    X_val = load_array(VAL + 'X_val.bc/')
    y_val = load_array(VAL + 'y_val.bc/')

    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    y_val[y_val == 0] = -1

    logging.info('Training Size ' + str(X_train.shape[0]))
    logging.info('Testing Size ' + str(X_test.shape[0]))
    logging.info('Val Size ' + str(X_val.shape[0]))

    moe = MOE()
    moe.tf_log = 'tf_log/log_seq_500'
    moe.C = config['C']
    moe.gamma = config['gamma']
    moe.iters = config['iters']
    moe.experts = config['experts']
    moe.gater_epoch = config['gater_epoch']
    moe.batch_size = config['batch_size']
    moe.hidden_units = config['hidden_units']
    moe.SEED = config['SEED']

    moe.train_dim = X_train.shape
    moe.test_dim = X_test.shape
    moe.val_dim = X_val.shape

    max_values = max_model(X_train)
    X_train = max_transform(max_values, X_train)
    X_test = max_transform(max_values, X_test)
    X_val = max_transform(max_values, X_val)

    logging.info('Iter, Train Error, Validation Error, Test Error')
    moe.train_model(X_train, y_train, X_test, y_test, X_val, y_val)

    total_time = time.time() - start_time
    logging.info("Time Taken " + str(total_time / 60))

    for i in moe.warn_log:
        logging.warning(i)

    logging.info('##### EXPERIMENT COMPLETE #####')
