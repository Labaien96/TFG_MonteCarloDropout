from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import keras.backend as K
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc
from IPython.display import clear_output
from keras.callbacks import CSVLogger, ModelCheckpoint

def do_stuff():
        num_classes = 2
        train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
                "/home/jokin/PycharmProjects/TFG/dogvscat/train/",
                target_size=(64, 64),
                batch_size=64,
                class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
                "/home/jokin/PycharmProjects/TFG/dogvscat/valid/",
                target_size=(64, 64),
                batch_size=64,
                class_mode='categorical')

        inputs = Input(shape=(64, 64, 3))

        conv1 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        drop10 = keras.layers.Dropout(0.3)(conv1)
        conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(drop10)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
        drop1 = keras.layers.Dropout(0.3)(pool1, training=True)

        conv3 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(drop1)
        drop20 = keras.layers.Dropout(0.3)(conv3)
        conv4 = keras.layers.Conv2D(128, (3, 3), activation='relu')(drop20)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
        drop2 = keras.layers.Dropout(0.3)(pool2, training=True)

        flat = Flatten()(drop2)
        dense1 = keras.layers.Dense(1024, activation='relu')(flat)
        drop3 = keras.layers.Dropout(0.3)(dense1, training=True)
        output = keras.layers.Dense(num_classes, activation='softmax')(drop3)

        model = Model(inputs=inputs, outputs=output)
        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        model.summary()
        csv_logger = CSVLogger('FAST_AI_03.csv', append=True, separator=';')
        filepath = "FAST_AI_03.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        model.fit_generator(train_generator,
                steps_per_epoch=23000//64,
                epochs=200,
                validation_data=validation_generator,
                validation_steps=2000//64,callbacks=[csv_logger,checkpoint],max_queue_size=10,
                workers=10,
                use_multiprocessing=True)

if __name__ == "__main__":
    gpu_device = "/gpu:0"
    if keras.backend.backend() == 'tensorflow':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device.rsplit(':', 1)[-1]
        session_config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        session = K.tf.Session(config=session_config)
        with K.tf.device(gpu_device):
            do_stuff()
