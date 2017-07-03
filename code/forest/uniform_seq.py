#!/usr/bin/env python3
import sys

sys.path.append('../helper_utils')
sys.path.append('/home/kkalyan3/code/helper_utils')
import time
from sklearn.utils import shuffle
from utils import load_array, max_model, max_transform
from sklearn.svm import SVC
import logging
import numpy as np
from sklearn.metrics import accuracy_score


class UniformSvm(object):

    def __init__(self):
        self.uniform_experts = 50
        self.cache_size = 50000
        self.C = 10
        self.cache_size = 50000
        self.gamma = 6
        self.train_dim = None
        self.test_dim = None
        self.val_dim = None
        self.experts = None

    def svc_model(self, X, y):
        X, y = shuffle(X, y)
        clf = SVC(C=self.C, kernel='rbf', gamma=self.gamma,
                  cache_size=self.cache_size, verbose=True, probability=True)
        model = clf.fit(X, y)
        return model

    def svc_eval(self, model, X, y):
        y_hat = model.predict(X)
        return accuracy_score(y, y_hat)

    def get_random(self):
        local_expert = {}
        random_bucket = np.random.choice(self.experts, self.train_dim[0])
        for i, e in enumerate(random_bucket):
            if e not in local_expert:
                local_expert[e] = [i]
            else:
                local_expert[e].append(i)
        return local_expert


    def train_model(self, x_train, y_train, x_test, y_test, x_val, y_val):

        split_buckets = self.get_random()

        y_hat_train = 0
        y_hat_test = 0
        y_hat_val = 0
        for key in sorted(split_buckets):
            X = x_train[split_buckets[key]]
            y = y_train[split_buckets[key]]
            model = self.svc_model(X, y)
            y_hat_train += model.predict(x_train)
            y_hat_test += model.predict(x_test)
            y_hat_val += model.predict(x_val)

        y_hat_train *= (1/self.experts)
        y_hat_test *= (1 / self.experts)
        y_hat_val *= (1 / self.experts)

        train_error = (1 - accuracy_score(y_train, y_hat_train > 0.5)) * 100
        test_error = (1 - accuracy_score(y_test, y_hat_test > 0.5)) * 100
        val_error = (1 - accuracy_score(y_val, y_hat_val > 0.5)) * 100

        return train_error, val_error, test_error

if __name__ == '__main__':
    logging.basicConfig(filename='../log/forest_uniform_seq.txt', level=logging.INFO)
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

    max_values = max_model(x_train)
    x_train = max_transform(max_values, x_train)
    x_test = max_transform(max_values, x_test)
    x_val = max_transform(max_values, x_val)

    usvm = UniformSvm()
    usvm.train_dim = x_train.shape
    usvm.test_dim = x_test.shape
    usvm.val_dim = x_val.shape
    usvm.experts = 50
    logging.info('C : {}, gamma : {}'.format(usvm.C, usvm.gamma))

    train_error, val_error, test_error = usvm.train_model(x_train, y_train, x_test, y_test, x_val, y_val)
    total_time = (time.time() - start_time) / 60

    logging.info('Train Error, Validation Error, Test Error, Time Taken')
    logging.info('{}, {}, {}, {}'.format(train_error, val_error, test_error, total_time))

    logging.info('##### EXPERIMENT COMPLETE #####')
