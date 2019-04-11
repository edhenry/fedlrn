import tensorflow as tf
import numpy as np
import time
import random


class DataLoader(object):
    """Generic dataloading object"""

    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = 5
    NUM_CLASSES = 10

    def __init__(self):

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        if tf.keras.backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = tf.keras.utils.to_categorical(y_train, DataLoader.NUM_CLASSES)
        self.y_test = tf.keras.utils.to_categorical(y_test, DataLoader.NUM_CLASSES)


class GlobalModel(object):
    """Generic GlobalModel object
    Used for locally testing model we're attempting
    to use in a federated learning context"""

    def __init__(self):
        self.model = self.build_model()

        self.current_weights = self.model.get_weights()
        self.previoud_training_loss = None

        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []

        self.training_start_time = int(round(time.time()))

    def build_model(self):
        raise NotImplementedError()


class CNN(GlobalModel):

    BATCH_SIZE = 128
    NUM_CLASSES = 10
    EPOCHS = 20

    def __init__(self):
        super(CNN, self).__init__()
        
    def build_model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                         activation='relu',
                                         input_shape=(28, 28, 1), ))
        model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        return model

if __name__ == "__main__":
    data = DataLoader()

    x_train, y_train = data.x_train, data.y_train
    x_test, y_test = data.x_test, data.y_test

    cnn = CNN()

    cnn_fit = cnn.model.fit(x_train, y_train,
                     batch_size=cnn.BATCH_SIZE,
                     epochs=cnn.EPOCHS,
                     verbose=1)

    print(cnn.model.summary())
    
    score = cnn.model.evaluate(x_test, y_test)
    print(f"Test loss : {score[0]}")
    print(f"Test accuracy : {score[1]}")