from __future__ import print_function
import keras
from keras import regularizers
from vgg_modif import VGG16
from keras.datasets import cifar10
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input,GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot as plt
from tqdm import tqdm
import keras.backend as K
import seaborn as sns
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from IPython.display import clear_output
from keras.callbacks import CSVLogger, ModelCheckpoint
import json
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.applications.vgg16 import preprocess_input
from keras.applications.imagenet_utils import _preprocess_numpy_input, _obtain_input_shape, decode_predictions
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs

from pprint import pprint
import PIL

from PIL import Image
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from skimage.io import imread

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

crop_length = 224

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]

def VGG16(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Dropout(0.4)(x,training=True)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    #x = Dropout(0.5)(x,training=True)


    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Dropout(0.4)(x,training=True)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    #x = Dropout(0.5)(x,training=True)


    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Dropout(0.4)(x,training=True)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Dropout(0.4)(x,training=True)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    #x = Dropout(0.5)(x,training=True)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Dropout(0.4)(x,training=True)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Dropout(0.4)(x,training=True)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #x = Dropout(0.5)(x,training=True)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Dropout(0.4)(x,training=True)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Dropout(0.4)(x,training=True)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    #x = Dropout(0.5)(x,training=True)
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)

    return model


'''
def crop_generator(batches, crop_length):

    Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator

    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)
'''


def do_stuff():
    with open('/mnt/RAID5/datasets/private/plant-disease/dataset_all/merged/ALLCROPS/image_list.json') as json_data:
        d = json.load(json_data)

    imagenes_trigo = []
    labels = []
    for i in tqdm(d):
        if d[i]["crop"] == "TRZAW":
            image_trig = '/mnt/RAID5/datasets/private/plant-disease/dataset_all/merged/ALLCROPS' + \
                         d[i]["diseases"]["SEPTTR"]["image_path"]
            # open_im = Image.open(image_trig)
            # open_im = open_im.resize((256,256), Image.ANTIALIAS)
            if i[len(i) - 4:len(i)] != ".JPG" and i[len(i) - 4:len(i)] != ".jpg":
                b = i + ".jpg"
            else:
                b = i
            xxx = os.path.join('trigo_resize', b)
            # open_im.save(xxx)
            imagenes_trigo.append(xxx)
            if d[i]["diseases"]["SEPTTR"]["disease_presence"] == True:
                labels.append([0, 1])
            else:
                labels.append([1, 0])

    x_train, x_test, y_train, y_test = train_test_split(imagenes_trigo, labels, test_size=0.25)

    class PlantasTrigoDataset(Sequence):
        def __init__(self, imagenes_trigo, labels):

            self.x_pre = []
            for i in imagenes_trigo:
                s = image.load_img(i, target_size=(256, 256))
                x = image.img_to_array(s)
                x = preprocess_input(x, data_format='channels_last', mode='caffe')
                self.x_pre.append(x)
            self.x = np.array(self.x_pre)
            '''
            #self.x=np.array([imread(i) for i in tqdm(imagenes_trigo)])
            self.x = np.array([imread(i) for i in tqdm(imagenes_trigo)])
            '''
            self.y = np.array(labels)
            self.batch_size = 4

        def __len__(self):
            return int(np.ceil(self.x.shape[0] / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
            for i in range(batch_x.shape[0]):
                batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
            return batch_crops, batch_y
            # np.array([file_name for file_name in batch_x]), np.array(batch_y)

        def get_steps(self):
            return self.x.shape[0] // self.batch_size

    data_train = PlantasTrigoDataset(x_train, y_train)
    # data_train = _preprocess_numpy_input(data_train, data_format='channels_first', mode = 'tf')
    data_val = PlantasTrigoDataset(x_test, y_test)
    # data_val = _preprocess_numpy_input(data_val, data_format='channels_first', mode='tf')
    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # show class indices
    # model = VGG_16('vgg16_weights.h5')

    # HOLA IDATZITA MODELO GUZTIA AGERTZEN JAT

    inputs = (224, 224, 3)
    model_sincabeza = VGG16(include_top=False, weights='imagenet', input_tensor=None,
                                                     input_shape=inputs, pooling=None, classes=2)

    fc1 = Dense(256, activation='relu', name='fc1')
    fc2 = Dense(256, activation='relu', name='fc2')
    predictions = Dense(2, activation='softmax', name='predictions')
    dropout1 = Dropout(0.4)
    dropout2 = Dropout(0.4)
    last_layer = model_sincabeza.layers[-1]
    x = Flatten(name='flatten')(last_layer.output)
    x = fc1(x)
    x = dropout1(x,training=True)
    x = fc2(x)
    x = dropout2(x,training=True)
    predictors = predictions(x)
    model_cabeza = Model(input=model_sincabeza.input, output=predictors)
    for layer in model_cabeza.layers[:19]:
        layer.trainable = False

    sgd = SGD(lr=1e-6, decay=1e-6, momentum=0.9, nesterov=True)
    model_cabeza.compile(loss='categorical_crossentropy',
                         optimizer=sgd,
                         metrics=['accuracy'])
    model_cabeza.summary()
    csv_logger = CSVLogger('PLANTS_04.csv', append=True, separator=';')
    filepath = 'PLANTS_04.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model_cabeza.fit_generator(
        data_train,
        steps_per_epoch=data_train.get_steps(),
        epochs=100,
        validation_data=data_val, validation_steps=data_val.get_steps(),
        callbacks=[csv_logger, checkpoint],
        max_queue_size=10,
        workers=6,
        use_multiprocessing=True)

    '''
    filepath = 'pretrain_plants2reg.hdf5'
    for layer in model_cabeza.layers:
        layer.trainable = True
    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)

    model_cabeza.load_weights(filepath)
    model_cabeza.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model_cabeza.summary()

    csv_logger = CSVLogger('train_plants2reg.csv', append=True, separator=';')
    filepath = 'train_plants2reg.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model_cabeza.fit_generator(
        data_train,
        steps_per_epoch=data_train.get_steps(),
        epochs=500,
        validation_data=data_val, validation_steps=data_val.get_steps(),
        callbacks=[csv_logger, checkpoint],
        max_queue_size=10,
        workers=6,
        use_multiprocessing=True)

    '''
    '''
    #MODELUA IDAZTEKO BESTE MODU BAT

    inputs = (224, 224, 3)
    model_sincabeza = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=inputs, pooling=None, classes=2)
    # Classification block
    my_input = Input(batch_shape=model_sincabeza.output_shape)
    x = Flatten(name='flatten')(my_input)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.25)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.25)(x)
    my_output = Dense(2, activation='softmax', name='predictions')(x)
    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    modelo_cabeza = Model(inputs=my_input, outputs=my_output,name='top_model')
    block5_pool = model_sincabeza.get_layer('block5_pool').output
    full_output = modelo_cabeza(block5_pool)
    full_model = Model(inputs=model_sincabeza.input, outputs=full_output)
    for layer in full_model.layers[:18]:
        layer.trainable = False

    full_model.summary()

    full_model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
    full_model.summary()

    csv_logger = CSVLogger('Pretrain_plants_tf_tf.csv', append=True, separator=';')
    filepath = 'Weights_pretrain_plants_tf_tf.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    full_model.fit_generator(
                    data_train,
                    steps_per_epoch=data_train.get_steps(),
                    epochs=50,
                    validation_data=data_val,validation_steps=data_val.get_steps(),callbacks=[csv_logger,checkpoint])


    inputs = (224, 224, 3)
    model_sincabeza = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=inputs, pooling=None, classes=2)
    # Classification block
    my_input = Input(batch_shape=model_sincabeza.output_shape)
    x = Flatten(name='flatten')(my_input)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.25)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.25)(x)
    my_output = Dense(2, activation='softmax', name='predictions')(x)
    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    modelo_cabeza = Model(inputs=my_input, outputs=my_output,name='top_model')
    block5_pool = model_sincabeza.get_layer('block5_pool').output
    full_output = modelo_cabeza(block5_pool)
    full_model2 = Model(inputs=model_sincabeza.input, outputs=full_output)
    full_model2.summary()

    full_model2.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
    full_model2.summary()
    #full_model2.load_weights('Weights_pretrain_plants_tf_tf.hdf5')
    csv_logger = CSVLogger('No_finetuning_plants.csv', append=True, separator=';')
    filepath = "No_finetuning_plants.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    full_model2.fit_generator(
        data_train,
        steps_per_epoch=data_train.get_steps(),
        epochs=500,
        validation_data=data_val,
        validation_steps=data_val.get_steps(), callbacks=[csv_logger, checkpoint])


    '''
    '''
    #MODELUA ENTRENAU, PREENTRENAU GABE

    input2 = Input(shape=(224, 224, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input2)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)


    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Top
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    output_drop = Dense(2, activation='softmax', name='predictions')(x)
    model = Model(inputs=input2,outputs=output_drop)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.load_weights('pretrain_plants2.hdf5')
    model.summary()
    csv_logger = CSVLogger('PLANTS.csv', append=True, separator=';')
    filepath = "PLANTS.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.fit_generator(
        data_train,
        steps_per_epoch=data_train.get_steps(),
        epochs=500,
        validation_data=data_val, validation_steps=data_val.get_steps(),
        callbacks=[csv_logger, checkpoint],
        max_queue_size=10,
        workers=10,
        use_multiprocessing=True)
    '''
    '''
    #AITORREK PASAUTAKO MODELUA

    model = VGG_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    csv_logger = CSVLogger('plants_AITOR.csv', append=True, separator=';')
    filepath = "weights_pretrain_AITOR.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    for layer in model.layers:
        layer.trainable = False
    model.layers[-1].trainable = True
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    model.fit_generator(
        data_train,
        steps_per_epoch=data_train.get_steps(),
        epochs=2,
        validation_data=data_val,
        validation_steps=data_val.get_steps(),
        callbacks=[csv_logger, checkpoint])

    model = VGG_16(filepath)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    csv_logger = CSVLogger('plants_AITOR_train.csv', append=True, separator=';')
    filepath = "weights_train_AITOR.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    model.fit_generator(
        data_train,
        steps_per_epoch=data_train.get_steps(),
        epochs=500,
        validation_data=data_val,
        validation_steps=data_val.get_steps(),
        callbacks=[csv_logger, checkpoint])
    '''
    '''
    input21 = Input(shape=(254, 254, 3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input2)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Top
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    output_drop1 = Dense(2, activation='softmax', name='predictions')(x)
    model2 = Model(inputs=input21, outputs=output_drop1)
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model2.load_weights(filepath)
    model2.compile(loss='categorical_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy'])
    model2.summary()
    for layer in model2.layers[:19]:
        layer.trainable = False

    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    model2.compile(loss='categorical_crossentropy',
                         optimizer=sgd,
                         metrics=['accuracy'])
    model2.summary()

    csv_logger = CSVLogger('pretrain_plants2.csv', append=True, separator=';')
    filepath = 'pretrain_plants2.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model2.fit_generator(
        data_train,
        steps_per_epoch=data_train.get_steps(),
        epochs=50,
        validation_data=data_val, validation_steps=data_val.get_steps(),
        callbacks=[csv_logger, checkpoint],
        max_queue_size=10,
        workers=6,
        use_multiprocessing=True)

    '''


if __name__ == "__main__":
    gpu_device = "/gpu:0"
    if keras.backend.backend() == 'tensorflow':
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device.rsplit(':', 1)[-1]
        session_config = K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_config.gpu_options.allow_growth = True
        session = K.tf.Session(config=session_config)
        with K.tf.device(gpu_device):
            do_stuff()

