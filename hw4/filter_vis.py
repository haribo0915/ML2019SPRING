import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K

# from marcos import *
train_path = sys.argv[1]
img_row, img_col = 48, 48
def main(): 
    raw_data = pd.read_csv(train_path)
    x_train = raw_data['feature'].values
    x_train = np.array([row.split(' ') for row in x_train], dtype=np.float32)
    #x_train = x_train / 255
    x_train = x_train.reshape(x_train.shape[0], img_row, img_col, 1)

    #data_name = args.data
    model_name = './model.h5'
    filter_dir = sys.argv[2]

    print('load model')
    emotion_classifier = load_model(model_name)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)

    print('load data') 
    X = x_train


    input_img = emotion_classifier.input
    name_ls = ['conv2d_3']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    choose_id = 100
    for cnt, fn in enumerate(collect_layers):
        photo = X[choose_id].reshape(1, 48, 48, 1)
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        #nb_filter = im[0].shape[3]
        nb_filter = 64
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/8, 8, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='OrRd')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('filter {}'.format(i))
            plt.tight_layout()
        #fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        img_path = os.path.join(filter_dir, 'fig2_2.jpg')
        fig.savefig(img_path)
        break

if __name__ == '__main__':
    main()