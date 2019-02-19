'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

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
import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)


def do_stuff():
    num_classes = 2
    # The data, split between train and test sets:
    (x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()
    print("SIZE X TRAIN", x_train1.shape, "SIZE Y TRAIN", y_train1.shape)
    y_train = y_train1[np.logical_or(y_train1 == 6, y_train1 == 7)]
    x_train = x_train1[np.logical_or(y_train1 == 6, y_train1 == 7).flatten()]
    y_test = y_test1[np.logical_or(y_test1 == 6, y_test1 == 7)]
    x_test = x_test1[np.logical_or(y_test1 == 6, y_test1 == 7).flatten()]
    for i in range(y_train.shape[0]):
        if y_train[i] == 6:
            y_train[i] = 0
        else:
            y_train[i] = 1
    for i in range(y_test.shape[0]):
        if y_test[i] == 6:
            y_test[i] = 0
        else:
            y_test[i] = 1

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    inputs = Input(shape=(32, 32, 3))

    conv1 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    drop10 = keras.layers.Dropout(0.25)(conv1)
    conv2 = keras.layers.Conv2D(32, (3, 3), activation='relu')(drop10)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop1 = keras.layers.Dropout(0.25)(pool1, training=True)

    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(drop1)
    drop20 = keras.layers.Dropout(0.25)(conv3)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu')(drop20)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop2 = keras.layers.Dropout(0.25)(pool2, training=True)

    flat = Flatten()(drop2)
    dense1 = keras.layers.Dense(512, activation='relu')(flat)
    drop3 = keras.layers.Dropout(0.25)(dense1, training=True)
    output = keras.layers.Dense(num_classes, activation='softmax')(drop3)

    model = Model(inputs=inputs, outputs=output)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.load_weights("weights.bestxVSy.hdf5")
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    inputs = Input(shape=(32, 32, 3))

    conv1 = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    drop10 = keras.layers.Dropout(0.25)(conv1)
    conv2 = keras.layers.Conv2D(32, (3, 3), activation='relu')(drop10)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop1 = keras.layers.Dropout(0.25)(pool1)

    conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(drop1)
    drop20 = keras.layers.Dropout(0.25)(conv3)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu')(drop20)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop2 = keras.layers.Dropout(0.25)(pool2)

    flat = Flatten()(drop2)
    dense1 = keras.layers.Dense(512, activation='relu')(flat)
    drop3 = keras.layers.Dropout(0.25)(dense1)
    output = keras.layers.Dense(num_classes, activation='softmax')(drop3)

    model2 = Model(inputs=inputs, outputs=output)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model2.load_weights("weights.bestxVSy.hdf5")
    # Let's train the model using RMSprop
    model2.compile(loss='categorical_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    result = np.zeros((200, 2000, num_classes))

    for i in tqdm(range(200)):
        result[i, :, :] = model.predict(x_test, verbose=0)

    uncertainty = np.zeros((2000))

    for i in range(2000):
        uncertainty[i] = result[:, i, 1].std(axis=0)

    y_predict = np.zeros((2000, num_classes))

    for i in range(2000):
        y_predict[i, 0] = result[:, i, 0].mean(axis=0)
        y_predict[i, 1] = result[:, i, 1].mean(axis=0)

    print("Y_PREDICT:", y_predict)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, threshold = roc_curve(y_test[:, 1], y_predict[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(0)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve CIFAR2')
    plt.legend(loc="lower right")

    y_predict2 = model2.predict(x_test, verbose=0)
    fpr2, tpr2, threshold2 = roc_curve(y_test[:, 1], y_predict2[:, 1])
    roc_auc2 = auc(fpr2, tpr2)

    uncertainty2 = []
    for i in range(2000):
        max = np.argmax(y_predict2[i])
        uncertainty2.append((1 - (y_predict2[i][max] - 0.5)) * 2)

    uncertainty_correct = []
    uncertainty_incorrect = []
    correct_predicted = []
    incorrect_predicted = []

    for i in range(2000):
        maximun = np.argmax(y_predict[i])
        prediction = np.unravel_index(maximun, y_predict[i].shape)
        maximun2 = np.argmax(y_test[i])
        real = np.unravel_index(maximun2, y_test[i].shape)
        if prediction == real:
            correct_predicted.append(i)
            uncertainty_correct.append(uncertainty[i])

        else:
            incorrect_predicted.append(i)
            uncertainty_incorrect.append(uncertainty[i])

    plt.figure(2)
    plt.xlim(0, 0.3)
    sns.kdeplot(uncertainty_correct[:len(uncertainty_incorrect)], label="Uncertainty of well clasificated images",
                color="g", shade=True);
    sns.kdeplot(uncertainty_incorrect, color="r", label="Uncertainty of incorrect clasificated images", shade=True);
    plt.legend()

    plt.figure(3)
    plt.hist(uncertainty_correct[:len(uncertainty_incorrect)], 30, label="Uncertainty of well clasificated images",
             density=True, facecolor='g', alpha=0.75)
    plt.grid(True)
    plt.hist(uncertainty_incorrect, 30, label="Uncertainty of incorrect clasificated images", density=True,
             facecolor='r', alpha=0.75)
    plt.grid(True)
    plt.legend()

    # score = model.evaluate(x_test, y_test, verbose=0)

    # plot_retained=[score[1]]
    plot_retained = [roc_auc]

    x_test_mc = x_test
    y_test_mc = y_test

    for i in tqdm(range(1999)):
        max_uncertain = np.argmax(uncertainty)
        x_test_mc = np.delete(x_test_mc, max_uncertain, 0)
        y_test_mc = np.delete(y_test_mc, max_uncertain, 0)
        y_predict = np.delete(y_predict, max_uncertain, 0)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        fpr, tpr, _ = roc_curve(y_test_mc[:, 1], y_predict[:, 1])
        roc_auc = auc(fpr, tpr)
        uncertainty = np.delete(uncertainty, max_uncertain, 0)

        plot_retained.append(roc_auc)

    plot_retained2 = [roc_auc2]
    x_test2 = x_test
    y_test2 = y_test
    for i in tqdm(range(1999)):
        max_uncertain = np.argmax(uncertainty2)
        x_test2 = np.delete(x_test2, max_uncertain, 0)
        y_test2 = np.delete(y_test2, max_uncertain, 0)
        y_predict2 = np.delete(y_predict2, max_uncertain, 0)

        fpr2 = dict()
        tpr2 = dict()
        roc_auc2 = dict()

        fpr2, tpr2, _ = roc_curve(y_test2[:, 1], y_predict2[:, 1])
        roc_auc2 = auc(fpr2, tpr2)
        uncertainty2 = np.delete(uncertainty2, max_uncertain, 0)

        plot_retained2.append(roc_auc2)

    plt.figure(1)

    plt.plot(plot_retained[::-1])
    plt.plot(plot_retained2[::-1], color="r")
    plt.ylabel('AUC')
    plt.xlabel('Retained data')

    plt.show()


if __name__ == "__main__":
    gpu_device = "/gpu:0"
    if keras.backend.backend() == 'tensorflow':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device.rsplit(':', 1)[-1]
        session_config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        session = K.tf.Session(config=session_config)
        with K.tf.device(gpu_device):
            do_stuff()
