#!/usr/bin/env python3
# TODO
# Use Prime
# Use data augmentation on CIFAR and SVHN
# Subset of Classes per expert (5 classes per expert)
# (Error code output correction)
# Prime Network Gater with Last Layer


import sys

sys.path.append('../helper_utils')
sys.path.append('/home/kkalyan3/code/helper_utils')
from utils import load_array, eval_target, ohe_decode
from nn_arch import nn_models
from keras.utils import np_utils
import numpy as np
import logging
import time
from keras.layers import Dense, Input, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from helper_callbacks import CustomCallback
import pandas as pd

PATH = '/home/kkalyan3/data/mnist/'
#PATH = '/gel/usr/skkal1/SVM-Experts/data/mnist/'
#PATH = '../../data/mnist/'


class MOE(object):
    def __init__(self):
        self.experts = None
        self.train_dim = None
        self.test_dim = None
        self.iters = 1
        self.target = 10
        self.wm_xi = None
        self.epoch = 1000
        self.c = 1
        self.early_stopping = CustomCallback()
        self.warn_log = [["Iter", "Expert Training Error", "Expert Val Error", "Expert Test Error"]]
        self.expert_keys = None

    def init_labels(self):
        import pandas as pd

        ecoc = {0: {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1},
                1: {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0, 8: 1, 9: 0},
                2: {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 0, 9: 1},
                3: {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1, 9: 0},
                4: {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1},
                5: {0: 1, 1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 1, 7: 0, 8: 1, 9: 0},
                6: {0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 1, 7: 0, 8: 0, 9: 1},
                7: {0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1, 9: 0},
                8: {0: 1, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1},
                9: {0: 1, 1: 0, 2: 1, 3: 0, 4: 0, 5: 1, 6: 1, 7: 0, 8: 1, 9: 0}}

        subset_data = pd.DataFrame.from_dict(ecoc)
        expert_keys = {}

        for i in range(10):
            current_col = subset_data.iloc[:, i]
            var_name = str('e') + str(i)
            labels = np.where(current_col == 1)
            np.random.shuffle(labels[0])
            expert_keys[var_name] = labels[0]
            print('{} Subsets Produced'.format(expert_keys))
        return expert_keys

    def split_buckets(self, iter, expert, y):
        expert = 'e'+str(expert)
        current_expert = self.expert_keys[expert]

        if iter == 0:
            y_decode = ohe_decode(y)
            # print(itemfreq(y_decode).astype(int))
            index = []
            # i <- [1,2,3,4,5]
            for i in current_expert:
                current_index = np.where(y_decode == i)[0]
                index.append(current_index)
        else:
            comp_key = 'w_' + str(iter) + '_' + str(expert)
            index = self.get_expert(self.wm_xi, current_expert, comp_key)

        return np.hstack(index)

    def get_expert(self, weight_data, expert_key, name):
        # import pdb;pdb.set_trace()
        index = []
        sort_index = np.argsort(-1 * weight_data)
        # write_csv(sort_index, name)

        for k, v in enumerate(sort_index):
            if v[0] in expert_key:
                index.append(v)
        return index

    def tensor_product(self, x):
        a = x[0]
        b = x[1]
        b = K.reshape(b, (-1, self.experts, self.target))
        y = K.batch_dot(b, a, axes=1)
        return y

    def gater(self):
        dim_inputs_data = Input(shape=(64,))
        dim_mlp_yhat = Input(shape=(self.target * self.experts,))

        layer_1 = Dense(50, activation='relu')(dim_inputs_data)
        layer_1_a = Dropout(0.5)(layer_1)
        layer_2 = Dense(50, activation='relu')(layer_1_a)
        layer_2_a = Dropout(0.5)(layer_2)
        layer_3 = Dense(self.experts, name='layer_op', activation='relu', use_bias=False)(layer_2_a)
        layer_4 = Lambda(self.tensor_product)([layer_3, dim_mlp_yhat])
        layer_5 = Dense(10, activation='softmax')(layer_4)
        model = Model(inputs=[dim_inputs_data, dim_mlp_yhat], outputs=layer_5)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def model_train(self, X, y, x_test, y_test, X_val, y_val, i):
        model = nn_models()
        model.ip_shape = X.shape
        model = model.lenet5()
        model.learning_rate = 0.0001
        model.fit(X, y, batch_size=256, epochs=self.epoch, validation_data=(X_val, y_val),
                  verbose=1, callbacks=[self.early_stopping])

        yhat_train = model.predict(X, batch_size=256)
        yhat_val = model.predict(x_val, batch_size=256)
        yhat_test = model.predict(x_test, batch_size=256)

        train_error = eval_target(yhat_train, y)
        val_error = eval_target(yhat_val, y_val)
        test_error = eval_target(yhat_test, y_test)

        self.warn_log.append([i, train_error, val_error, test_error])

        return model

    def main(self, x_train, y_train, x_test, y_test, x_val, y_val):

        print("############################# Prime Train ################################")
        model_p = nn_models()
        model_p.ip_shape = x_train.shape
        model_p.learning_rate = 0.0001
        model_p = model_p.lenet5()
        model_p.fit(x_train, y_train, batch_size=256, epochs=self.epoch, validation_data=(x_val, y_val),
                    verbose=1, callbacks=[self.early_stopping])

        model_prime = Model(inputs=model_p.input,
                            outputs=model_p.get_layer('dense2').output)

        prime_op_tr = model_prime.predict(x_train)
        prime_op_tt = model_prime.predict(x_test)
        prime_op_v = model_prime.predict(x_val)

        prime_op_train = model_p.predict(x_train)
        prime_op_val = model_p.predict(x_val)
        prime_op_test = model_p.predict(x_test)

        prime_train_e = eval_target(prime_op_train, y_train)
        prime_val_e = eval_target(prime_op_val, y_val)
        prime_test_e = eval_target(prime_op_test, y_test)

        self.warn_log.append([-1, prime_train_e, prime_val_e, prime_test_e])

        for i in range(self.iters):
            yhat_train_exp = []
            yhats_test_exp = []
            yhats_val_exp = []
            for j in range(self.experts):
                print("############################# Expert {} Iter {} ################################".format(j, i))

                buckets = self.split_buckets(i, j, y_train)
                X = x_train[buckets]
                y = y_train[buckets]

                model = self.model_train(X, y, x_test, y_test, x_val, y_val, i)

                yhat_train = model.predict(x_train, batch_size=256)
                yhats_test = model.predict(x_test, batch_size=256)
                yhats_val = model.predict(x_val, batch_size=256)

                yhat_train_exp.append(yhat_train)
                yhats_test_exp.append(yhats_test)
                yhats_val_exp.append(yhats_val)

            yhat_tr = np.hstack(yhat_train_exp)
            yhat_tt = np.hstack(yhats_test_exp)
            yhat_val = np.hstack(yhats_val_exp)

            model = self.gater()
            history = model.fit([prime_op_tr, yhat_tr], y_train, shuffle=True,
                                batch_size=256, verbose=1,
                                validation_data=([prime_op_v, yhat_val], y_val),
                                epochs=self.epoch, callbacks=[self.early_stopping])

            yhats_train = model.predict([prime_op_tr, yhat_tr], batch_size=256)
            yhats_test = model.predict([prime_op_tt, yhat_tt], batch_size=256)
            yhats_val = model.predict([prime_op_v, yhat_val], batch_size=256)

            tre = eval_target(yhats_train, y_train)
            tte = eval_target(yhats_test, y_test)
            vale = eval_target(yhats_val, y_val)

            logging.info('{}, {}, {}, {}'.format(i, tre, vale, tte))

            expert_units = Model(inputs=model.input,
                                 outputs=model.get_layer('layer_op').output)

            self.wm_xi = expert_units.predict([prime_op_tr, yhat_tr])

        return None


if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(filename='../log/mnist_subset.' + str(start_time), level=logging.INFO)

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

    img_rows, img_cols = 28, 28

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    y_val = np_utils.to_categorical(y_val, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')

    x_train /= 255
    x_test /= 255
    x_val /= 255

    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    x_val -= mean_image

    # Pre process end
    gater = MOE()
    gater.experts = 10
    gater.train_dim = x_train.shape
    gater.test_dim = x_test.shape
    gater.expert_keys = gater.init_labels()

    logging.info("Experts " + str(gater.experts))
    logging.info("Train " + str(x_train.shape))
    logging.info("Val " + str(x_val.shape))
    logging.info("Test " + str(x_test.shape))

    logging.info('{}, {}, {}, {}'.format("Iter", "Training Error", "Val Error", "Test Error"))
    gater.main(x_train, y_train, x_test, y_test, x_val, y_val)

    total_time = (time.time() - start_time) / 60
    logging.info("Total Time " + str(total_time))

    for i in gater.warn_log:
        logging.warning(i)

    logging.info("#### Experiment Complete ####")
