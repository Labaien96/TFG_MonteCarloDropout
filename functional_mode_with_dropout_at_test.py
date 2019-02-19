from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np

batch_size = 32
num_classes = 2
epochs = 8
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()
print("SIZE X TRAIN",x_train1.shape,"SIZE Y TRAIN",y_train1.shape)
y_train=y_train1[np.logical_or(y_train1==3, y_train1==5)]
x_train=x_train1[np.logical_or(y_train1==3, y_train1==5).flatten()]
y_test=y_test1[np.logical_or(y_test1==3, y_test1==5)]
x_test=x_test1[np.logical_or(y_test1==3, y_test1==5).flatten()]
for i in range(y_train.shape[0]):
    if y_train[i]==3:
        y_train[i] = 0
    else:
        y_train[i] = 1
for i in range(y_test.shape[0]):
    if y_test[i]==3:
        y_test[i] = 0
    else:
        y_test[i] = 1
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print("SIZE X TRAIN",x_train.shape,"SIZE Y TRAIN",y_train.shape)
# Convert class vectors to binary class matrices.
print("Y_TRAIN",y_train)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
a = Input(shape=(x_train.shape[1:],))
b = Conv2D(x_train.shape[1:],(3,3),padding='same')(a)
model = Model(inputs=a, outputs=b)