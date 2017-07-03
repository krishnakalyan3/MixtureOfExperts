#!/usr/bin/env python3

import time
from helper_utils import load_array, scale_model, max_model, max_transform
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import logging
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV


INSTANCES = 100000

class SVM(object):
    def __init__(self):
        self.sample_size = 3500
        self.params = {}
        self.cache_size = 40000
        self.verbose = 1
        self.iters = 5000

    def svc_model(self, X, y):
        X, y = shuffle(X, y, random_state=1337)
        svc = SVC(kernel='rbf', cache_size=self.cache_size, verbose=True)
        clf = RandomizedSearchCV(svc, param_distributions=self.params, n_iter=self.iters, n_jobs=-1, verbose=self.verbose)
        model = clf.fit(X[0:self.sample_size], y[0:self.sample_size])
        logging.info('Grid Scores ' + str(model.best_params_))
        logging.info('Best Scores ' + str(model.best_score_))
        return model.best_estimator_

    def svc_eval(self, model, X, y):
        y_hat = model.predict(X)
        return (1 - accuracy_score(y, y_hat)) * 100

if __name__ == '__main__':
    logging.basicConfig(filename='../log/random_sigma.txt', level=logging.INFO)
    start_time = time.time()

    logging.info('##### NEW EXPERIMENT #####')

    TRAIN = '../data/train/' + str(100) + '/'
    TEST = '../data/test/' + str(100) + '/'
    VAL = '../data/val/' + str(100) + '/'

    X_train = load_array(TRAIN + 'X_train.bc/')
    y_train = load_array(TRAIN + 'y_train.bc/')
    X_test = load_array(TEST + 'X_test.bc/')
    y_test = load_array(TEST + 'y_test.bc/')
    X_val = load_array(VAL + 'X_val.bc/')
    y_val = load_array(VAL + 'y_val.bc/')

    max_values = max_model(X_train)
    X_train = max_transform(max_values, X_train)
    X_test = max_transform(max_values, X_test)
    logging.info('Scaling Finished')

    svm = SVM()
    svm.params['C'] = np.logspace(-4, 6, svm.iters)
    svm.params['gamma'] = np.logspace(-5, 6, svm.iters)
    svc_model = svm.svc_model(X_train[0:INSTANCES], y_train[0:INSTANCES])
    logging.info('Model Training Complete')

    train_accuracy = svm.svc_eval(svc_model, X_train, y_train)
    test_accuracy = svm.svc_eval(svc_model, X_test, y_test)
    print("Train ", train_accuracy)
    print("Test ", test_accuracy)
    logging.info('Train Error ' + str(train_accuracy))
    logging.info('Test Error ' + str(test_accuracy))
    total_time = time.time() - start_time
    logging.info("Time Taken " + str(total_time/60))
    logging.info('##### EXPERIMENT COMPLETE #####')
