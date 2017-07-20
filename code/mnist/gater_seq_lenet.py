#!/usr/bin/env python3

import sys
sys.path.append('../helper_utils')
sys.path.append('/home/kkalyan3/code/helper_utils')
import logging
import time
import numpy as np
from keras.utils import np_utils
from helper_callbacks import CustomCallback
from nn_arch import nn_models
from utils import load_array, eval_target
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda
from keras import backend as K

PATH = '/home/kkalyan3/data/mnist/'
#PATH = '/gel/usr/skkal1/SVM-Experts/data/mnist/'
#PATH = '../../data/mnist/'

class Gater(object):
    def __init__(self):
        self.experts = None
        self.train_dim = None
        self.test_dim = None
        self.iters = 6
        self.wm_xi = None
        self.target = 10
        self.early_stopping = CustomCallback()
        self.c = 1
        self.warn_log = [["Iter", "Expert Training Error", "Expert Val Error"]]

    def get_expert(self, weight_data):
        thresh_hard = int(weight_data.shape[0] / weight_data.shape[1]) + self.c

        # [0.2, 0.1, 0.7, 0.1] -> [2, 0 ,1, 3]
        sort_index = np.argsort(-1 * weight_data)
        thresh_dict = {}

        # {0:0 , 1:0 , 2:0 ..}
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

    def train_model(self, X, y, x_val, y_val, i):
        model = nn_models()
        model.ip_shape = X.shape
        model = model.lenet5()
        model.fit(X, y, batch_size=256, epochs=500, validation_data=(x_val, y_val),
                  verbose=1, callbacks=[self.early_stopping])

        yhat_train = model.predict(X, batch_size=256)
        yhat_val = model.predict(x_val, batch_size=256)

        train_error = eval_target(yhat_train, y)
        val_error = eval_target(yhat_val, y_val)
        self.warn_log.append([i, train_error, val_error])

        return model

    def tensor_product(self, x):
        a = x[0]
        b = x[1]
        b = K.reshape(b, (-1, self.experts, self.target))
        y = K.batch_dot(b, a, axes=1)
        return y

    def gater(self):
        dim_inputs_data = Input(shape=(64, ))
        dim_mlp_yhat = Input(shape=(self.target * self.experts,))

        layer_1 = Dense(50, activation='relu')(dim_inputs_data)
        layer_1_a = Dropout(0.5)(layer_1)
        layer_2 = Dense(50, activation='relu')(layer_1_a)
        layer_2_a = Dropout(0.5)(layer_2)
        layer_3 = Dense(self.experts, name='layer_op', activation='relu', use_bias=False)(layer_2_a)
        layer_4 = Lambda(self.tensor_product)([layer_3, dim_mlp_yhat])
        layer_5 = Dense(10, activation='softmax')(layer_4)

        model = Model(inputs=[dim_inputs_data, dim_mlp_yhat], outputs=layer_5)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def main(self, x_train, y_train, x_test, y_test, x_val, y_val):
        print("############################# Prime Train ################################")
        model_prime = nn_models()
        model_prime.ip_shape = x_train.shape
        model_p = model_prime.lenet5()

        model_prime = Model(inputs=model_p.input,
                            outputs=model_p.get_layer('dense2').output)

        prime_op_tr = model_prime.predict(x_train)
        prime_op_tt = model_prime.predict(x_test)
        prime_op_v = model_prime.predict(x_val)

        for i in range(self.iters):
            split_buckets = self.bucket_function(i)

            experts_out_train = []
            experts_out_test = []
            experts_out_val = []
            for j in sorted(split_buckets):
                X = x_train[split_buckets[j]]
                y = y_train[split_buckets[j]]

                model = self.train_model(X, y, x_val, y_val, i)
                yhats_train = model.predict(x_train, batch_size=256)
                yhats_test = model.predict(x_test, batch_size=256)
                yhats_val = model.predict(x_val, batch_size=256)

                experts_out_train.append(yhats_train)
                experts_out_test.append(yhats_test)
                experts_out_val.append(yhats_val)

            yhat_tr = np.hstack(experts_out_train)
            yhat_tt = np.hstack(experts_out_test)
            yhat_val = np.hstack(experts_out_val)

            model = self.gater()
            history = model.fit([prime_op_tr, yhat_tr], y_train, shuffle=True,
                                batch_size=256, verbose=1,
                                validation_data=([prime_op_v, yhat_val], y_val),
                                epochs=500, callbacks=[self.early_stopping])

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
    logging.basicConfig(filename='../log/mnist_gater_seq_lenet' + str(start_time), level=logging.INFO)
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

    gater = Gater()
    gater.experts = 10
    gater.train_dim = x_train.shape
    gater.test_dim = x_test.shape
    logging.info("Experts {}".format(gater.experts))

    logging.info('{}, {}, {}, {}'.format("Training Error", "Val Error", "Test Error", "Time"))
    gater.main(x_train, y_train, x_test, y_test, x_val, y_val)

    total_time = (time.time() - start_time) / 60
    logging.info("Total Time {}".format(total_time))

    for i in gater.warn_log:
        logging.warning(i)

    logging.info("#### Experiment Complete ####")
