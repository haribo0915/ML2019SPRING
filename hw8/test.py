import keras
import pandas as pd
import numpy as np
from keras.optimizers import *
from keras.models import Model, load_model
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
import sys

test_path = sys.argv[1]
out_path = sys.argv[2]
img_row, img_col = 48, 48

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

if __name__ == '__main__':
    raw = pd.read_csv(test_path)
    x_test = raw['feature'].values
    x_test = np.array([row.split(' ') for row in x_test], dtype=np.float32)

    x_test /= 255
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

    input_shape = (48, 48, 1)
    model = MobileNet(input_shape, 7)
    weight = np.load('compressed_weights.npy')
    model.set_weights(weight)

    y_test = model.predict(x_test)
    y = np.zeros(y_test.shape[0])
    for i in range(y_test.shape[0]):
        y[i] = int(np.argmax(y_test[i]))

    print(y)

    pd.DataFrame([[i, int(y[i])] for i in range(y.shape[0])], columns=['id', 'label']) \
            .to_csv(out_path, index=False)