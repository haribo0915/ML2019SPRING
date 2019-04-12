import numpy as np
import pandas as pd
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
import os 
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

raw_data = pd.read_csv(train_path)
x_train = raw_data['feature'].values
x_train = np.array([row.split(' ') for row in x_train], dtype=np.float32)
y_train = raw_data['label'].values
x_train = x_train / 255
x_train = x_train.reshape(x_train.shape[0], img_row, img_col, 1)

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

batch_size = 128
epochs = 400
input_shape = (48,48,1)
model = Sequential()
model.add(Conv2D(64,input_shape=input_shape, kernel_size=(5, 5), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.35))

model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU(alpha=1./20))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

early_stop = EarlyStopping(monitor='val_acc', patience=15, verbose=1)
check_save  = ModelCheckpoint("model/model1-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)

train_history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch = (5*len(x_train)//batch_size), epochs=epochs, verbose=1, validation_data=(x_valid, y_valid),
                        callbacks=[check_save, early_stop])


score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test accuracy:', score[1])

model.save(sys.argv[2])