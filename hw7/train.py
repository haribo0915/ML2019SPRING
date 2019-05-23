import numpy as np
from skimage import io 
import sys
import os
image_path = sys.argv[1]

x_train = []
for i in range(40000):
	s = str(i+1).zfill(6)
    name = s + '.jpg'
	img = io.imread(os.path.join(image_path, name))
	x_train.append(img)
x_train = np.array(x_train, dtype=float)
x_train /= 255.0

from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Dense, Reshape, Conv2DTranspose, Flatten
from keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 128]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    return model

'''input_img = Input(shape=(32, 32, 3)) 

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.summary()
autoencoder.compile(optimizer=Adam(), loss='mse')'''
autoencoder = CAE(input_shape=(32,32,3), filters=[32, 64, 128, 32])
autoencoder.summary()
autoencoder.compile(optimizer=Adam(), loss='mse')

callbacks = [ModelCheckpoint('./model/{epoch:02d}_{loss:.4f}_{val_loss:.4f}.h5', period=5)]
autoencoder.fit(x_train, x_train, batch_size=128, epochs=50,
          verbose=1, callbacks=callbacks, validation_split=0.1, shuffle=True)
model.save('model.h5')