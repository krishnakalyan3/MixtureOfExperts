from keras.callbacks import Callback
import numpy as np


class CustomCallback(Callback):
    def on_train_begin(self, logs={}):
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.val_loss.append(logs.get('val_loss'))

        if len(self.val_loss) > 100:
            last_10 = np.mean(self.val_loss[-10:])
            last_50 = np.mean(self.val_loss[-50:])
            if last_10 > last_50:
                print('\n Epoch %05d: early stopping ' % epoch)
                self.model.stop_training = True
