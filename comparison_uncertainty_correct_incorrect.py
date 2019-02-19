from __future__ import print_function
from keras.models import load_model
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
from keras.callbacks import CSVLogger


def do_stuff():
    correct_predicted = []
    incorrect_predicted = []

    batch_size = 32
    num_classes = 2
    epochs = 200
    data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    # The data, split between train and test sets:
    (x_train1, y_train1), (x_test1, y_test1) = cifar10.load_data()
    print("SIZE X TRAIN", x_train1.shape, "SIZE Y TRAIN", y_train1.shape)
    y_train = y_train1[np.logical_or(y_train1 == 3, y_train1 == 5)]
    x_train = x_train1[np.logical_or(y_train1 == 3, y_train1 == 5).flatten()]
    y_test = y_test1[np.logical_or(y_test1 == 3, y_test1 == 5)]
    x_test = x_test1[np.logical_or(y_test1 == 3, y_test1 == 5).flatten()]
    for i in range(y_train.shape[0]):
        if y_train[i] == 3:
            y_train[i] = 0
        else:
            y_train[i] = 1
    for i in range(y_test.shape[0]):
        if y_test[i] == 3:
            y_test[i] = 0
        else:
            y_test[i] = 1
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print("SIZE X TRAIN", x_train.shape, "SIZE Y TRAIN", y_train.shape)
    # Convert class vectors to binary class matrices.
    print("Y_TRAIN", y_train)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = load_model('keras_cifar10_trained_model.h5')
    # since it is a binary classification problem, the uncertainty for dogs and cats is the same, so I will compute only one
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
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

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
    plt.legend(loc="lower right")

    uncertainty_correct = []
    uncertainty_incorrect = []
    for i in range(2000):
        maximun = np.argmax(y_predict[i])
        prediction = np.unravel_index(maximun, y_predict[i].shape)
        maximun2 = np.argmax(y_test[i])
        real = np.unravel_index(maximun2, y_test[i].shape)
        if prediction == real:
            correct_predicted.append(result[:, i, 0])
            uncertainty_correct.append(uncertainty[i])
        else:
            incorrect_predicted.append(result[:, i, 0])
            uncertainty_incorrect.append(uncertainty[i])

    plt.figure()
    plt.hist(uncertainty_correct[:len(uncertainty_incorrect)], 30, density=True, facecolor='g', alpha=0.75)
    plt.grid(True)
    plt.hist(uncertainty_incorrect, 30, density=True, facecolor='b', alpha=0.75)
    plt.grid(True)
    plt.show()

    print("UNCERTAINTY IN CORRECT PREDICTIONS:", uncertainty_correct)


if __name__ == "__main__":
    gpu_device = "/gpu:0"
    if keras.backend.backend() == 'tensorflow':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device.rsplit(':', 1)[-1]
        session_config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        session = K.tf.Session(config=session_config)
        with K.tf.device(gpu_device):
            do_stuff()
