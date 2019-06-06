import numpy as np
import pandas as pd
import keras
import sys
from keras.optimizers import *
from keras.models import Model, load_model
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_path = sys.argv[1]
img_row, img_col = 48, 48

def shuffle_split_data(X, y, percent):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, percent)
    X_train = X[split]
    y_train = y[split]
    X_test =  X[~split]
    y_test = y[~split]

    return X_train, y_train, X_test, y_test

def get_conv_block(tensor, channels, strides, alpha=1.0, name=''):
    x = Conv2D(channels,
               kernel_size=(3, 3),
               strides=strides,
               use_bias=False,
               padding='same',
               name='{}_conv'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn'.format(name))(x)
    x = Activation('relu', name='{}_act'.format(name))(x)
    return x


def get_dw_sep_block(tensor, channels, strides, alpha=1.0, name=''):
    """Depthwise separable conv: A Depthwise conv followed by a Pointwise conv."""
    channels = int(channels * alpha)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        use_bias=False,
                        padding='same',
                        name='{}_dw'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn1'.format(name))(x)
    x = Activation('relu', name='{}_act1'.format(name))(x)

    # Pointwise
    x = Conv2D(channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=False,
               padding='same',
               name='{}_pw'.format(name))(x)
    x = BatchNormalization(name='{}_bn2'.format(name))(x)
    x = Activation('relu', name='{}_act2'.format(name))(x)
    return x


def MobileNet(shape, num_classes, alpha=1, include_top=True, weights=None):
    x_in = Input(shape=shape)

    x = get_conv_block(x_in, 16, (2, 2), alpha=alpha, name='initial')

    layers = [
        (16, (1, 1)),
        (32, (1, 1)),
        (64, (1, 1)),
        (128, (1, 1)),
        (128, (1, 1)),
        (256, (1, 1))
    ]

    for i, (channels, strides) in enumerate(layers):
        x = get_dw_sep_block(x, channels, strides, alpha=alpha, name='block{}'.format(i))

    if include_top:
        x = GlobalAvgPool2D(name='global_avg')(x)
        x = Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=x_in, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model

raw_data = pd.read_csv(train_path)
x_train = raw_data['feature'].values
x_train = np.array([row.split(' ') for row in x_train], dtype=np.float32)
y_train = raw_data['label'].values
x_train = x_train / 255
x_train = x_train.reshape(x_train.shape[0], img_row, img_col, 1)

batch_size = 128
epochs = 200
input_shape = (48,48,1)

datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.8, 1.2],
            shear_range=0.2,
            horizontal_flip=True) 

x_train, y_train, x_valid, y_valid = shuffle_split_data(x_train, y_train, percent=85)
y_train = keras.utils.to_categorical(y_train, 7)
y_valid = keras.utils.to_categorical(y_valid, 7)

data_iter = datagen.flow(x_train, y_train, batch_size=batch_size)

model = MobileNet(input_shape, 7)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#print(model.summary())

early_stop = EarlyStopping(monitor='val_acc', patience=15, verbose=1)
check_save  = ModelCheckpoint("model-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)

train_history = model.fit_generator(data_iter,
                        steps_per_epoch = (5*len(x_train)//batch_size), epochs=epochs, verbose=1, validation_data=(x_valid, y_valid),
                        callbacks=[check_save, early_stop])

wei = model.get_weights()
for i in range(len(wei)):
    wei[i] = wei[i].astype(np.float16)
np.save('compressed_weights.npy', wei)