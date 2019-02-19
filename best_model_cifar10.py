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
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc
from IPython.display import clear_output
from keras.callbacks import CSVLogger, ModelCheckpoint


# PLOT OF LOSSES AFTER EVERY EPOCH


def do_stuff():
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
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

    model.load_weights("weights.best_cifar10_drop3.hdf5")
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

    model2.load_weights("weights.best_cifar10_drop3.hdf5")
    # Let's train the model using RMSprop
    model2.compile(loss='categorical_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    result = np.zeros((200, 10000, num_classes))
    for i in tqdm(range(200)):
        result[i, :, :] = model.predict(x_test, verbose=0)

    uncertainty0 = np.zeros((10000))
    uncertainty1 = np.zeros((10000))
    uncertainty2 = np.zeros((10000))
    uncertainty3 = np.zeros((10000))
    uncertainty4 = np.zeros((10000))
    uncertainty5 = np.zeros((10000))
    uncertainty6 = np.zeros((10000))
    uncertainty7 = np.zeros((10000))
    uncertainty8 = np.zeros((10000))
    uncertainty9 = np.zeros((10000))

    for i in range(2000):
        uncertainty0[i] = result[:, i, 0].std(axis=0)
        uncertainty1[i] = result[:, i, 1].std(axis=0)
        uncertainty2[i] = result[:, i, 2].std(axis=0)
        uncertainty3[i] = result[:, i, 3].std(axis=0)
        uncertainty4[i] = result[:, i, 4].std(axis=0)
        uncertainty5[i] = result[:, i, 5].std(axis=0)
        uncertainty6[i] = result[:, i, 6].std(axis=0)
        uncertainty7[i] = result[:, i, 7].std(axis=0)
        uncertainty8[i] = result[:, i, 8].std(axis=0)
        uncertainty9[i] = result[:, i, 9].std(axis=0)

    y_predict = np.zeros((2000, num_classes))

    for i in range(2000):
        y_predict[i, 0] = result[:, i, 0].mean(axis=0)
        y_predict[i, 1] = result[:, i, 1].mean(axis=0)
        y_predict[i, 2] = result[:, i, 2].mean(axis=0)
        y_predict[i, 3] = result[:, i, 3].mean(axis=0)
        y_predict[i, 4] = result[:, i, 4].mean(axis=0)
        y_predict[i, 5] = result[:, i, 5].mean(axis=0)
        y_predict[i, 6] = result[:, i, 6].mean(axis=0)
        y_predict[i, 7] = result[:, i, 7].mean(axis=0)
        y_predict[i, 8] = result[:, i, 8].mean(axis=0)
        y_predict[i, 9] = result[:, i, 9].mean(axis=0)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

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
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    clear_output(wait=True)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve CIFAR2')
    plt.savefig("roc_cifar10.png")
    plt.legend(loc="lower right")

    plt.figure(3)
    plt.xlim(0, 0.3)
    sns.kdeplot(uncertainty_correct[:len(uncertainty_incorrect)], label="Uncertainty of well clasificated images",
                color="g", shade=True);
    sns.kdeplot(uncertainty_incorrect, color="r", label="Uncertainty of incorrect clasificated images", shade=True);
    plt.savefig("comparison_uncertainties_cifar10.png")

    plt.legend()



    x_test_mc = x_test
    y_test_mc = y_test
    score = model.evaluate(x_test, y_test, verbose=0)
    plot_retained = [score[1]]
    for i in tqdm(range(1500)):
        max_uncertain = np.argmax(uncertainty)
        x_test_mc = np.delete(x_test_mc, max_uncertain, 0)
        y_test_mc = np.delete(y_test_mc, max_uncertain, 0)
        y_predict = np.delete(y_predict, max_uncertain, 0)
        score = model.evaluate(x_test, y_test, verbose=0)
        uncertainty = np.delete(uncertainty, max_uncertain, 0)
        plot_retained.append(score[1])

    x_test2 = x_test
    y_test2 = y_test
    score2 = model2.evaluate(x_test2, y_test2, verbose=0)
    plot_retained = [score2[1]]
    for i in tqdm(range(1500)):
        max_uncertain = np.argmax(uncertainty2)
        x_test2 = np.delete(x_test2, max_uncertain, 0)
        y_test2 = np.delete(y_test2, max_uncertain, 0)
        y_predict2 = np.delete(y_predict2, max_uncertain, 0)
        uncertainty2 = np.delete(uncertainty2, max_uncertain, 0)
        score2 = model2.evaluate(x_test2, y_test2, verbose=0)
        plot_retained2.append(score2[1])



    y_predict4 = model2.predict(x_test, verbose=0)
    score3 = model2.evaluate(x_test, y_test, verbose=0)
    plot_retained4 = [score3[1]]
    x_test4 = x_test
    y_test4 = y_test

    for i in tqdm(range(1500)):
        x_test4 = np.delete(x_test4, 1999 - i, 0)
        y_test4 = np.delete(y_test4, 1999 - i, 0)
        y_predict4 = np.delete(y_predict4, 1999 - i, 0)
        plot_retained4.append(roc_auc4)

    plt.figure(1)
    plt.plot(plot_retained[::-1], label="MC Dropout")
    plt.plot(plot_retained2[::-1], color="r", label="Normal dropout")
    plt.plot(plot_retained4[::-1], color="g", label="Random")
    plt.ylabel('AUC')
    plt.xlabel('Retained data')
    plt.savefig("retainedMCvsSMAXvsRandom.png")
    plt.legend()

    plt.figure(1)
    plt.xlim(0, 0.4)
    sns.kdeplot(uncertainty0, label="Uncertainty class 0",
                color="g", shade=True);
    sns.kdeplot(uncertainty1, label="Uncertainty class 1",
                color="r", shade=True);
    sns.kdeplot(uncertainty2, label="Uncertainty class 2",
                color="aqua", shade=True);
    sns.kdeplot(uncertainty3, label="Uncertainty class 3",
                color="k", shade=True);
    sns.kdeplot(uncertainty4, label="Uncertainty class 4",
                color="yellow", shade=True);
    sns.kdeplot(uncertainty5, label="Uncertainty class 5",
                color="y", shade=True);
    sns.kdeplot(uncertainty6, label="Uncertainty class 6",
                color="m", shade=True);
    sns.kdeplot(uncertainty7, label="Uncertainty class 7",
                color="c", shade=True);
    sns.kdeplot(uncertainty8, label="Uncertainty class 8",
                color="seagreen", shade=True);
    sns.kdeplot(uncertainty9, label="Uncertainty class 9",
                color="tan", shade=True);

    plt.legend()
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
