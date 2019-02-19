from __future__ import print_function
import keras
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.applications.vgg16 import preprocess_input
from keras.applications.imagenet_utils import _preprocess_numpy_input
from vgg_modif import VGG16
from keras.datasets import cifar10
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
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
from keras.applications.imagenet_utils import _preprocess_numpy_input
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
crop_length   = 224

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
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
def VGG16_2(include_top=True, weights='imagenet',
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
    x = Dropout(0.4)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    #x = Dropout(0.5)(x,training=True)


    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    #x = Dropout(0.5)(x,training=True)


    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    #x = Dropout(0.5)(x,training=True)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #x = Dropout(0.5)(x,training=True)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Dropout(0.4)(x)
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
def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]
def do_stuff():
    with open('/mnt/RAID5/datasets/private/plant-disease/dataset_all/merged/ALLCROPS/image_list.json') as json_data:
        d = json.load(json_data)

    imagenes_trigo = []
    labels = []
    for i in tqdm(d):
        if d[i]["crop"] == "TRZAW":
            image_trig = '/mnt/RAID5/datasets/private/plant-disease/dataset_all/merged/ALLCROPS'+d[i]["diseases"]["SEPTTR"]["image_path"]
            #open_im = Image.open(image_trig)
            #open_im = open_im.resize((256,256), Image.ANTIALIAS)
            if i[len(i)-4:len(i)]!=".JPG" and i[len(i)-4:len(i)]!=".jpg":
                b=i+".jpg"
            else:
                b=i
            xxx = os.path.join('trigo_resize', b)
            #open_im.save(xxx)
            imagenes_trigo.append(xxx)
            if d[i]["diseases"]["SEPTTR"]["disease_presence"]==True:
                labels.append([0,1])
            else:
                labels.append([1,0])

    x_train,x_test,y_train,y_test = train_test_split(imagenes_trigo,labels,test_size=0.25)

    class PlantasTrigoDataset(Sequence):
        def __init__(self,imagenes_trigo,labels):

            self.x_pre = []
            for i in imagenes_trigo:
                s = image.load_img(i, target_size=(256, 256))
                x = image.img_to_array(s)
                x = preprocess_input(x,data_format='channels_last', mode='caffe')
                self.x_pre.append(x)
            self.x = np.array(self.x_pre)
            '''
            #self.x=np.array([imread(i) for i in tqdm(imagenes_trigo)])
            self.x = np.array([imread(i) for i in tqdm(imagenes_trigo)])
            '''
            self.y = np.array(labels)
            self.batch_size = 48
        def __len__(self):
            return int(np.ceil(self.x.shape[0]/float(self.batch_size)))
        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
            for i in range(batch_x.shape[0]):
                batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
            return batch_crops,batch_y
                #np.array([file_name for file_name in batch_x]), np.array(batch_y)
        def get_steps(self):
            return self.x.shape[0]//self.batch_size

    data_val = PlantasTrigoDataset(x_test,y_test)
    num_classes = 2
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
    model = Model(input=model_sincabeza.input, output=predictors)
    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    filepath = 'VGG16_256_04.hdf5'
    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()

    model_sincabeza = VGG16_2(include_top=False, weights='imagenet', input_tensor=None,
                                                     input_shape=inputs, pooling=None, classes=2)

    fc1 = Dense(256, activation='relu', name='fc1')
    fc2 = Dense(256, activation='relu', name='fc2')
    predictions = Dense(2, activation='softmax', name='predictions')
    dropout1 = Dropout(0.4)
    dropout2 = Dropout(0.4)
    last_layer = model_sincabeza.layers[-1]
    x = Flatten(name='flatten')(last_layer.output)
    x = fc1(x)
    x = dropout1(x)
    x = fc2(x)
    x = dropout2(x)
    predictors = predictions(x)
    model2 = Model(input=model_sincabeza.input, output=predictors)
    sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
    filepath = 'VGG16_256_04.hdf5'
    model2.load_weights(filepath)

    model2.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model2.summary()
    

    result = np.zeros((200, 5988, num_classes))

    for i in tqdm(range(200)):
        result[i, :, :] = model.predict_generator(data_val, verbose=0,max_queue_size=10,
                          workers=6,
                          use_multiprocessing=True)

    y_test = np.array(y_test)
    uncertainty = np.zeros((5988))

    for i in range(5988):
        uncertainty[i] = result[:, i, 1].std(axis=0)

    y_predict = np.zeros((5988, num_classes))

    for i in range(5988):
        y_predict[i, 0] = result[:, i, 0].mean(axis=0)
        y_predict[i, 1] = result[:, i, 1].mean(axis=0)

    y_predict2 = model2.predict_generator(data_val, verbose=0,max_queue_size=10,
                          workers=6,
                          use_multiprocessing=True)
    fpr2 = dict()
    tpr2 = dict()
    roc_auc2 = dict()
    fpr2, tpr2, threshold2 = roc_curve(y_test[:, 1], y_predict2[:, 1])
    roc_auc2 = auc(fpr2, tpr2)

    uncertainty2 = []
    for i in range(5988):
        max = np.argmax(y_predict2[i])
        uncertainty2.append((1 - (y_predict2[i][max] - 0.5)) * 2)

    uncertainty_correct = []
    uncertainty_incorrect = []
    correct_predicted = []
    incorrect_predicted = []

    for i in range(5988):
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
    df_cm = pd.DataFrame(conf_mat, index=[i for i in ["OSASUNTSU", "GAIXORIK"]],
                         columns=[i for i in ["OSASUNTSU", "GAIXORIK"]])
    plt.figure(8)
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.savefig('128_03_1.png')

    plt.figure(2)
    plt.xlim(0, 0.3)
    sns.kdeplot(uncertainty_correct[:len(uncertainty_incorrect)],
                label="Ondo klasifikatutako argazkien ziurgabetasuna",
                color="g", shade=True);
    sns.kdeplot(uncertainty_incorrect, color="r", label="Gaizki klasifikatutako argazkien ziurgabetasuna", shade=True);
    plt.legend()
    plt.savefig('128_03_2.png')

    plt.figure(3)
    plt.hist(uncertainty_correct[:len(uncertainty_incorrect)], 30,
             label="Ondo klasifikatutako argazkien ziurgabetasuna",
             density=True, facecolor='g', alpha=0.75)
    plt.grid(True)
    plt.hist(uncertainty_incorrect, 30, label="Gaizki klasifikatutako argazkien ziurgabetasuna", density=True,
             facecolor='r', alpha=0.75)
    plt.grid(True)
    plt.legend()
    plt.savefig('128_03_3.png')
    fpr, tpr, threshold = roc_curve(y_test[:, 1], y_predict[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(5)
    plt.plot(fpr, color='r')
    plt.plot(tpr, color='green')
    plt.savefig('128_03_4.png')
    plt.figure(4)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC kurba (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='green', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa positibo faltsua')
    plt.ylabel('Benetako tasa positiboa')
    plt.title('ROC kurba')
    plt.legend(loc="lower right")
    plt.savefig('128_03_5.png')
    plot_retained = [roc_auc2]
    x_test_mc = x_test
    y_test_mc = y_test
    plt.figure(6)
    plt.plot(threshold, fpr, color='r', label='Tasa positibo faltsua')
    plt.plot(threshold, tpr, color='green', label='Benetako tasa positiboa')
    plt.savefig('128_03_6.png')
    plt.legend()
    y_predict = y_predict2

    for i in tqdm(range(5900)):
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

    for i in tqdm(range(5900)):
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

    y_predict3 = model2.predict_generator(data_val, verbose=0, max_queue_size=10,
                          workers=6,
                          use_multiprocessing=True)
    fpr3, tpr3, threshold3 = roc_curve(y_test[:, 1], y_predict3[:, 1])
    roc_auc3 = auc(fpr3, tpr3)
    plot_retained3 = [roc_auc3]
    x_test3 = x_test
    y_test3 = y_test
    uncertainty_entropy = []
    for i in range(5988):
        uncertainty_entropy.append(
            -(y_predict3[i][1] * np.log(y_predict3[i][1]) + (1 - y_predict3[i][1]) * np.log(1 - y_predict3[i][1])))

    for i in tqdm(range(5900)):
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

    y_predict4 = model2.predict_generator(data_val, verbose=0,max_queue_size=10,
                          workers=6,
                          use_multiprocessing=True)
    fpr4, tpr4, threshold4 = roc_curve(y_test[:, 1], y_predict4[:, 1])
    roc_auc4 = auc(fpr4, tpr4)
    plot_retained4 = [roc_auc4]
    x_test4 = x_test
    y_test4 = y_test

    for i in tqdm(range(5900)):
        j = np.random.random_integers(5900 - i)
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
    plt.savefig('128_03_7.png')
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
