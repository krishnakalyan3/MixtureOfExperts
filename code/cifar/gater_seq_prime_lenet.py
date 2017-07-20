#!/usr/bin/env python3

# TODO:
# Train Data

import sys

sys.path.append('../helper_utils')
sys.path.append('/home/kkalyan3/code/helper_utils')
from utils import load_array, eval_target
from nn_arch import nn_models
from keras.utils import np_utils
import numpy as np
import logging
import time
from keras.layers import Dense, Input, Dropout, Lambda
from keras.models import Model
from keras import backend as K
from helper_callbacks import CustomCallback

PATH = '/home/kkalyan3/data/cifar10/'
#PATH = '../../data/cifar10/'

class MOE(object):
    def __init__(self):
        self.experts = None
        self.train_dim = None
        self.test_dim = None
        self.expert_dim = None
        self.iters = 4
        self.target = 10
        self.wm_xi = None
        self.c = 1
        self.early_stopping = CustomCallback()
        self.warn_log = [["Iter", "Expert Training Error", "Expert Val Error"]]

    def get_random(self):
        local_expert = {}
        random_bucket = np.random.choice(self.experts, self.train_dim[0])
        for i, e in enumerate(random_bucket):
            if e not in local_expert:
                local_expert[e] = [i]
            else:
                local_expert[e].append(i)
        return local_expert

    def model_train(self, X, y, X_val, y_val, i):
        model = nn_models()
        model.ip_shape = X.shape
        model = model.lenet5()
        model.fit(X, y, batch_size=256, epochs=500, validation_data=(X_val, y_val),
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

        layer_2 = Dense(150, activation='relu')(dim_inputs_data)
        layer_2_b = Dropout(0.5)(layer_2)
        layer_3 = Dense(self.experts, name='layer_op', activation='relu', use_bias=False)(layer_2_b)
        layer_4 = Lambda(self.tensor_product)([layer_3, dim_mlp_yhat])
        layer_4b = Dropout(0.5)(layer_4)
        layer_10 = Dense(10, activation='softmax')(layer_4b)
        model = Model(inputs=[dim_inputs_data, dim_mlp_yhat], outputs=layer_10)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def bucket_function(self, i):
        if i == 0:
            return self.get_random()
        else:
            return self.get_expert(self.wm_xi)

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

    def main(self, x_train, y_train, x_test, y_test, x_val, y_val):

        model_prime = nn_models()
        model_prime.ip_shape = x_train.shape
        model_p = model_prime.lenet5()

        model_p.fit(x_train, y_train, batch_size=256, epochs=500, validation_data=(x_val, y_val),
                  verbose=1, callbacks=[self.early_stopping])

        model_prime = Model(inputs=model_p.input,
                             outputs=model_p.get_layer('dense2').output)

        prime_op_tr = model_prime.predict(x_train)
        prime_op_tt = model_prime.predict(x_test)
        prime_op_v = model_prime.predict(x_val)

        prime_op_train = model_p.predict(x_train)
        prime_op_val = model_p.predict(x_val)
        tre = eval_target(prime_op_train, y_train)
        vale = eval_target(prime_op_val, y_val)

        self.warn_log.append([-1, tre, vale])

        for i in range(self.iters):
            split_buckets = self.bucket_function(i)
            yhat_train_exp = []
            yhats_test_exp = []
            yhats_val_exp = []
            for expert_index in sorted(split_buckets):

                y = y_train[split_buckets[expert_index]]
                X = x_train[split_buckets[expert_index]]
                model = self.model_train(X, y, x_val, y_val, i)

                yhat_train = model.predict(x_train, batch_size=256)
                yhats_test = model.predict(x_test, batch_size=256)
                yhats_val = model.predict(x_val, batch_size=256)

                yhat_train_exp.append(yhat_train)
                yhats_test_exp.append(yhats_test)
                yhats_val_exp.append(yhats_val)

                print("Expert Index {}".format(expert_index))

            yhat_tr = np.hstack(yhat_train_exp)
            yhat_tt = np.hstack(yhats_test_exp)
            yhat_val = np.hstack(yhats_val_exp)

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

        return "Gater Training Complete"

if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(filename='../log/cifair_gater_simple_prime_lenet'+str(start_time), level=logging.INFO)

    logging.info('##### NEW EXPERIMENT_' + str(start_time) + '_#####')

    TRAIN = PATH + 'train/400/'
    TEST = PATH + 'test/400/'
    VAL = PATH + 'val/400/'

    x_train = load_array(TRAIN + 'x_train.bc/')
    y_train = load_array(TRAIN + 'y_train.bc/')
    x_test = load_array(TEST + 'x_test.bc/')
    y_test = load_array(TEST + 'y_test.bc/')
    x_val = load_array(VAL + 'x_val.bc/')
    y_val = load_array(VAL + 'y_val.bc/')

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
    uniform = MOE()
    uniform.experts = 10
    uniform.train_dim = x_train.shape
    uniform.test_dim = x_test.shape

    logging.info("Experts " + str(uniform.experts))
    logging.info("Train " + str(x_train.shape))
    logging.info("Val " + str(x_val.shape))
    logging.info("Test " + str(x_test.shape))

    logging.info('{}, {}, {}, {}'.format("Iter", "Training Error", "Val Error", "Test Error"))
    uniform.main(x_train, y_train, x_test, y_test, x_val, y_val)

    total_time = (time.time() - start_time)/60
    logging.info("Total Time " + str(total_time))

    for i in uniform.warn_log:
        logging.warning(i)

    logging.info("#### Experiment Complete ####")
