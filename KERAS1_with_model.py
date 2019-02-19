'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras import Input
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
from matplotlib import pyplot as plt
batch_size = 128
num_classes = 10
epochs = 1

# the data, split between train and test sets
(x_train1, y_train1), (x_test1, y_test1) = mnist.load_data() #SPLIT THE DATA IN TWO PARTS: TRAIN AND TEST
print(x_train1.shape) #(60.000,28,28)-> 60.000 IMAGES OF 28X28 PIXELS
print(x_test1.shape) #(10.000,28,28)-> 10.000 IMAGES OF 28X28 PIXELS
print(x_train1[0])

y_train=y_train1[np.logical_or(y_train1==0, y_train1==1)]
x_train=x_train1[np.logical_or(y_train1==0, y_train1==1).flatten()]
y_test=y_test1[np.logical_or(y_test1==0, y_test1==1)]
x_test=x_test1[np.logical_or(y_test1==0, y_test1==1).flatten()]


print("X_TRAIN:",x_train.shape) #(60.000,28,28)-> 60.000 IMAGES OF 28X28 PIXELS
print("X_TEST",x_test.shape) #(10.000,28,28)-> 10.000 IMAGES OF 28X28 PIXELS


x_train = x_train.reshape(12665, 784)
x_test = x_test.reshape(2115, 784)


x_train = x_train.astype('float32') # float32 FORMATUAN JARTZEN DITXU LISTAKO BALIXO GUZTIAK
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("after astype")
print(x_train.shape)
print(x_test.shape)
print(x_train)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices

print(y_train)
y_train = keras.utils.to_categorical(y_train, num_classes) #Converts a class vector (integers) to binary class matrix. Zenbaki bakoitza 0-1 balixuekin osatutako matrize bihurtzen dau.
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train)

inputs=Input(shape=(784,))
x=keras.layers.Dense(512,activation='relu')(inputs)
y=keras.layers.Dropout(0.2)(x, training=True)
j=keras.layers.Dense(512,activation='relu')(y)
k=keras.layers.Dropout(0.2)(j, training=True)
output=Dense(num_classes, activation='softmax')(k)

model=Model(inputs=inputs,outputs=output)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test)) 
#Epoch: number of times we will train the model
#Batch: How many samples we will take in each step
#Verbose=1 da ikusteko pausu bakoitzian zein dan loss function, accuracy, time... 
#score = model.evaluate(x_test, y_test, verbose=2)
result=np.zeros((100,2115,10))
for i in range(100):
    result[i,:,:]=model.predict(x_test, verbose=0)
prediction = result.mean(axis=0)
uncertainty = result.std(axis=0)
res2=result[:,0,1]
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
print("SHAPE PREDICTIONS AFTER 100 ITERATIONS",result.shape)
print("PREDICTIONS",res2)
print("MEAN",prediction.shape)
print("UNCERTAINTY",uncertainty.shape)
plt.hist(res2, 50, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Samples')
plt.ylabel('Probability')
plt.title('Histogram of 2nd class')
plt.axis([0, 1, 0, 1])
plt.grid(True)
plt.show()