
from keras.layers import Dense, Input, merge, Merge, Dropout, MaxPooling2D, Flatten, Conv2D,\
    AveragePooling2D, ZeroPadding2D, Lambda, Activation
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import optimizers
from keras.layers.normalization import BatchNormalization

class nn_models():
    def __init__(self):
        self.ip_shape = None
        self.learning_rate = 0.001

    def vgg_preprocess(self,x):
        x = x - self.vgg_mean
        return x[:, ::-1]

    # CIFAR ERROR - 21%
    def lenet5(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5,), name='conv1',
                         padding='same',
                                activation='relu',
                                input_shape=self.ip_shape[1:]))

        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
        # Local Normalization
        model.add(Conv2D(64, (5, 5,), padding='same', activation='relu', name='conv2'))
        # Local Normalization
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))

        model.add(Flatten())
        model.add(Dense(128, activation='relu', name='dense1'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', name='dense2'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax', name='dense3'))

        adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
        return model

    def simple_nn(self):
        model = Sequential()
        model.add(Conv2D(64, (self.stride, self.stride,), name='conv1',
                         padding='same',
                         activation='relu',
                         input_shape=self.ip_shape[1:]))

        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))

        model.add(Flatten())
        model.add(Dense(64, activation='relu', name='dense2'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax', name='dense3'))
        adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
        return model

    def cuda_cnn(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5),
                         border_mode='same',
                         activation='relu',
                         input_shape=self.ip_shape[1:]))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(contrast normalization)
        model.add(Conv2D(32, (5, 5), border_mode='valid', activation='relu'))
        model.add(AveragePooling2D(border_mode='same'))
        # model.add(contrast normalization)
        model.add(Conv2D(64, (5, 5), border_mode='valid', activation='relu'))
        model.add(AveragePooling2D(border_mode='same'))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
        return model

    def fat_mlp(self):
        model = Sequential()
        model.add(Dense(50, activation='tanh', input_shape=self.ip_shape[1:]))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
        return model

    def forest_mlp(self):
        model = Sequential()
        model.add(Dense(150, activation='relu', input_shape=self.ip_shape[1:]))
        model.add(Dropout(0.5))
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
        return model

    def mlp(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.ip_shape[1:]))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
        return model

    def small_nn(self):
        model = Sequential()
        model.add(Conv2D(64, (self.stride, self.stride,), name='conv1',
                         padding='same',
                         activation='relu',
                         input_shape=self.ip_shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(32, activation='relu', name='dense1'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax', name='dense2'))
        adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
        return model

    def small_nn_soft(self, temp):
        model = Sequential()
        model.add(Conv2D(64, (self.stride, self.stride,), name='conv1',
                         padding='same',
                         activation='relu',
                         input_shape=self.ip_shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(32, activation='relu', name='dense1'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10, name='dense2'))
        model.add(Lambda(lambda x: x / temp))
        model.add(Activation('softmax'))

        adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
        return model

    def big_nn(self):
        model = Sequential()
        model.add(Conv2D(64, (self.stride, self.stride,), name='conv1',
                         padding='same',
                         activation='relu',
                         input_shape=self.ip_shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
        model.add(BatchNormalization())

        model.add(Conv2D(32, (self.stride, self.stride,), name='conv2',
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
        model.add(BatchNormalization())

        model.add(Conv2D(16, (self.stride, self.stride,), name='conv3',
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(256, activation='relu', name='dense1'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax', name='dense2'))
        adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
        return model