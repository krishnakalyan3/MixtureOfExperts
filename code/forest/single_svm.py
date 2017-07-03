#!/usr/bin/env python3

import sys

sys.path.append('../helper_utils')
sys.path.append('/home/kkalyan3/code/helper_utils')
import time
from utils import load_array, max_model, max_transform
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import logging
from sklearn.utils import shuffle


class SVM(object):
    def __init__(self):
        self.C = 10
        self.cache_size = 5000
        self.gamma = 6
        self.max_val = None

    def svc_model(self, X, y):
        X, y = shuffle(X, y)
        clf = SVC(C=self.C, kernel='rbf', gamma=self.gamma, cache_size=self.cache_size,
                  verbose=True)
        model = clf.fit(X, y)
        return model

    def svc_eval(self, model, X, y):
        y_hat = model.predict(X)
        return (1 - accuracy_score(y, y_hat)) * 100

if __name__ == '__main__':
    logging.basicConfig(filename='../log/forest_single_svm.txt', level=logging.INFO)
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
    logging.info('Scaling Finished')

    svm = SVM()
    svc_model = svm.svc_model(x_train, y_train)
    logging.info('Model Training Complete')
    logging.info('C : {}, gamma : {}'.format(svm.C, svm.gamma))

    train_error = svm.svc_eval(svc_model, x_train, y_train)
    test_error = svm.svc_eval(svc_model, x_test, y_test)
    val_error = svm.svc_eval(svc_model, x_val, y_val)
    total_time = (time.time() - start_time)/60

    logging.info('Train Error, Validation Error, Test Error, Time Taken')
    logging.info('{}, {}, {}, {}'.format(train_error, val_error, test_error, total_time))

    logging.info('##### EXPERIMENT COMPLETE #####')
