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
import multiprocessing as mp
from functools import partial



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

    def svc_model(self, params):
        X, y = params
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

    @staticmethod
    def predict_models(X, model):
        return model.predict(X)

    def train_model(self, x_train, y_train, x_test, y_test, x_val, y_val):

        split_buckets = self.get_random()

        experts_out_train = np.empty((self.train_dim[0], self.uniform_experts))
        experts_out_test = np.empty((self.test_dim[0], self.uniform_experts))
        experts_out_val = np.empty((self.val_dim[0], self.uniform_experts))

        X = [x_train[split_buckets[k]] for k in sorted(split_buckets)]
        y = [y_train[split_buckets[j]] for j in sorted(split_buckets)]

        params1 = zip(X, y)
        models = pool1.map(self.svc_model, params1)

        func1 = partial(self.predict_models, x_train)
        func2 = partial(self.predict_models, x_test)
        func3 = partial(self.predict_models, x_val)

        train_yhat = pool1.map(func1, models)
        test_yhat = pool1.map(func2, models)
        val_yhat = pool1.map(func3, models)

        for k, _ in enumerate(models):
            experts_out_train[:, k] = train_yhat[k]
            experts_out_test[:, k] = test_yhat[k]
            experts_out_val[:, k] = val_yhat[k]

        train_error = (1 - accuracy_score(y_train, experts_out_train.mean(axis=1) > 0.5)) * 100
        test_error = (1 - accuracy_score(y_test, experts_out_test.mean(axis=1) > 0.5)) * 100
        val_error = (1 - accuracy_score(y_val, experts_out_val.mean(axis=1) > 0.5)) * 100

        return train_error, val_error, test_error

if __name__ == '__main__':
    pool1 = mp.Pool(processes=mp.cpu_count())
    logging.basicConfig(filename='../log/forest_uniform_par.txt', level=logging.INFO)
    start_time = time.time()

    logging.info('##### NEW EXPERIMENT #####')
    logging.info('Total CPUs : {}'.format(mp.cpu_count()))

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