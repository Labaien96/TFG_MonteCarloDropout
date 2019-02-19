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
import pandas as pd
from keras.backend import expand_dims
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
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

        validation_generator = test_datagen.flow_from_directory(
                "/home/jokin/PycharmProjects/TFG/dogvscat/valid/",
                target_size=(64, 64),
                batch_size=64,
                class_mode='categorical',shuffle=False)
        data_list = []
        label_list = []
        batch_index=0
        while batch_index<=validation_generator.batch_index:
            data=validation_generator.next()
            data_list.append(data[0])
            label_list.append(data[1])
            batch_index=batch_index+1
        data = []
        label = []
        for i in data_list:
            for j in i:
                data.append(j)
        for i in label_list:
            for j in i:
                label.append(j)
        x_test = np.asarray(data)
        y_test = np.asarray(label)


        inputs = Input(shape=(64, 64, 3))
        conv1 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        drop10 = keras.layers.Dropout(0.25)(conv1, training=True)
        conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(drop10)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
        drop1 = keras.layers.Dropout(0.25)(pool1, training=True)

        conv3 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(drop1)
        drop20 = keras.layers.Dropout(0.25)(conv3, training=True)
        conv4 = keras.layers.Conv2D(128, (3, 3), activation='relu')(drop20)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
        drop2 = keras.layers.Dropout(0.25)(pool2, training=True)

        flat = Flatten()(drop2)
        dense1 = keras.layers.Dense(1024, activation='relu')(flat)
        drop3 = keras.layers.Dropout(0.25)(dense1, training=True)
        output = keras.layers.Dense(num_classes, activation='softmax')(drop3)

        model = Model(inputs=inputs, outputs=output)
        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.load_weights("FAST_AI_03.hdf5")
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        inputs = Input(shape=(64, 64, 3))

        conv1 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        drop10 = keras.layers.Dropout(0.25)(conv1)
        conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu')(drop10)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
        drop1 = keras.layers.Dropout(0.25)(pool1)

        conv3 = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(drop1)
        drop20 = keras.layers.Dropout(0.25)(conv3)
        conv4 = keras.layers.Conv2D(128, (3, 3), activation='relu')(drop20)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
        drop2 = keras.layers.Dropout(0.25)(pool2)

        flat = Flatten()(drop2)
        dense1 = keras.layers.Dense(1024, activation='relu')(flat)
        drop3 = keras.layers.Dropout(0.25)(dense1)
        output = keras.layers.Dense(num_classes, activation='softmax')(drop3)

        model2 = Model(inputs=inputs, outputs=output)
        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model2.load_weights("FAST_AI_03.hdf5")
        model2.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])


        result = np.zeros((200, 2000, num_classes))
        #result = np.zeros((200, num_classes))
        x_test_one = np.expand_dims(x_test[0], axis=0)
        for i in tqdm(range(200)):
            #result[i, :] = model.predict(x_test_one, verbose=0)
            result[i, :, :] = model.predict_generator(validation_generator, verbose=0)

        res2 = result[:, 0, 1]
        res1 = result[:, 0, 0]
        plt.figure(3)
        plt.hist(res2, 30, density=True, facecolor='g', alpha=0.75)
        plt.xlabel('Score')
        plt.title('Txakurren histograma')
        plt.grid(True)

        print("RES2= ", res2)
        plt.figure(4)
        plt.hist(res1, 30, density=True, facecolor='r', alpha=0.75)
        plt.xlabel('Score')
        plt.title('Katuen histograma')
        plt.grid(True)
        plt.show()

        uncertainty = np.zeros((2000))

        for i in range(2000):
            uncertainty[i] = result[:, i, 1].std(axis=0)

        y_predict = np.zeros((2000, num_classes))

        for i in range(2000):
            y_predict[i, 0] = result[:, i, 0].mean(axis=0)
            y_predict[i, 1] = result[:, i, 1].mean(axis=0)

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

        yt = [np.argmax(t) for t in y_test]
        yp = [np.argmax(t) for t in y_predict]
        conf_mat = confusion_matrix(yt, yp)
        df_cm = pd.DataFrame(conf_mat, index=[i for i in ["TXAKURRA", "KATUA"]],
                             columns=[i for i in ["TXAKURRA", "KATUA"]])
        plt.figure(8)
        sns.heatmap(df_cm, annot=True, fmt='d')
        plt.savefig('fastai8.png')
        plt.figure(2)
        plt.xlim(0, 0.3)
        sns.kdeplot(uncertainty_correct[:len(uncertainty_incorrect)], label="Ondo klasifikatutako argazkien ziurgabetasuna",
                    color="g", shade=True);
        sns.kdeplot(uncertainty_incorrect, color="r", label="Gaizki klasifikatutako argazkien ziurgabetasuna", shade=True);
        plt.savefig('fastai7.png')
        plt.legend()

        plt.figure(3)
        plt.hist(uncertainty_correct[:len(uncertainty_incorrect)], 30, label="Ondo klasifikatutako argazkien ziurgabetasuna",
                 density=True, facecolor='g', alpha=0.75)
        plt.grid(True)
        plt.hist(uncertainty_incorrect, 30, label="Gaizki klasifikatutako argazkien ziurgabetasuna", density=True,
                 facecolor='r', alpha=0.75)
        plt.grid(True)
        plt.legend()
        plt.savefig('fastai1.png')
        fpr, tpr, threshold = roc_curve(y_test[:, 1], y_predict[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(5)
        plt.plot(fpr,color='r')
        plt.plot(tpr, color='green')
        plt.savefig('fastai6.png')
        plt.figure(4)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC kurba (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa positibo faltsua')
        plt.ylabel('Benetako tasa positiboa')
        plt.title('Fastai datu basearen ROC kurba')
        plt.legend(loc="lower right")
        plt.savefig('fastai2.png')
        plot_retained = [roc_auc]
        x_test_mc = x_test
        y_test_mc = y_test

        plt.figure(6)
        plt.plot(threshold,fpr,color='r',label='Tasa positibo faltsua')
        plt.plot(threshold,tpr, color='green', label='Benetako tasa positiboa')
        plt.savefig('fastai3.png')
        plt.legend()


        for i in tqdm(range(1900)):
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
        for i in tqdm(range(1900)):
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

        y_predict3 = model2.predict(x_test, verbose=0)
        fpr3, tpr3, threshold3 = roc_curve(y_test[:, 1], y_predict3[:, 1])
        roc_auc3 = auc(fpr3, tpr3)
        plot_retained3 = [roc_auc3]
        x_test3 = x_test
        y_test3 = y_test
        uncertainty_entropy = []
        for i in range(2000):
            uncertainty_entropy.append(
                -(y_predict3[i][1] * np.log(y_predict3[i][1]) + (1 - y_predict3[i][1]) * np.log(1 - y_predict3[i][1])))

        for i in tqdm(range(1900)):
            max_uncertain = np.argmax(uncertainty_entropy)
            x_test3 = np.delete(x_test3, max_uncertain, 0)
            y_test3 = np.delete(y_test3, max_uncertain, 0)
            y_predict3 = np.delete(y_predict3, max_uncertain, 0)

            fpr3 = dict()
            tpr3 = dict()
            roc_auc3 = dict()

            fpr3, tpr3, _ = roc_curve(y_test3[:, 1], y_predict3[:, 1])
            roc_auc3 = auc(fpr3, tpr3)
            uncertainty_entropy = np.delete(uncertainty_entropy, max_uncertain, 0)

            plot_retained3.append(roc_auc3)

        y_predict4 = model2.predict(x_test, verbose=0)
        fpr4, tpr4, threshold4 = roc_curve(y_test[:, 1], y_predict4[:, 1])
        roc_auc4 = auc(fpr4, tpr4)
        plot_retained4 = [roc_auc4]
        x_test4 = x_test
        y_test4 = y_test

        for i in tqdm(range(1900)):
            j =  np.random.randint(1900-i)
            x_test4 = np.delete(x_test4, j, 0)
            y_test4 = np.delete(y_test4, j, 0)
            y_predict4 = np.delete(y_predict4, j, 0)

            fpr4 = dict()
            tpr4 = dict()
            roc_auc4 = dict()

            fpr4, tpr4, _ = roc_curve(y_test4[:, 1], y_predict4[:, 1])
            roc_auc4 = auc(fpr4, tpr4)

            plot_retained4.append(roc_auc4)

        plt.figure(1)
        plt.plot(plot_retained[::-1], label="MC Dropout")
        plt.plot(plot_retained2[::-1], color="r", label="Dropout normala")
        plt.plot(plot_retained3[::-1], color="g", label="Entropia")
        plt.plot(plot_retained4[::-1], color="fuchsia", label="Ausazkoa")
        plt.ylabel('AUC')
        plt.xlabel('Argazki kopurua')
        plt.legend()
        plt.savefig('fastai4.png')
        plt.show()


'''
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
        plt.ylabel('Accuracy')
        plt.xlabel('Retained data')
        plt.legend()
        plt.show()
        plt.show()'''

if __name__ == "__main__":
    gpu_device = "/gpu:0"
    if keras.backend.backend() == 'tensorflow':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device.rsplit(':', 1)[-1]
        session_config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        session = K.tf.Session(config=session_config)
        with K.tf.device(gpu_device):
            do_stuff()
