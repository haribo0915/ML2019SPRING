import numpy as np
import pandas as pd
from skimage import io 
from time import time
import sys
import os

image_path = sys.argv[1]
test_path = sys.argv[2]
pred_path = sys.argv[3]

def savePrediction(y, path, id_start=0):
    y = np.array(y)
    pd.DataFrame([[i+id_start, y[i]] for i in range(y.shape[0])],
                 columns=['id', 'label']).to_csv(path, index=False)
def preprocess():
	image = []
	for i in range(40000):
		s = str(i+1).zfill(6)
		name = s + '.jpg'
		img = io.imread(os.path.join(image_path, name))
		image.append(img)
	image = np.array(image, dtype=float)
	image /= 255.0
	return image

start = time()

image = preprocess()	

from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, Dense, Reshape, Conv2DTranspose, Flatten
from keras.models import Model, load_model, Sequential

model = load_model('model.h5')
autoencoder = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)

image = autoencoder.predict(image)

print('Run TSNE....')
from MulticoreTSNE import MulticoreTSNE as TSNE
tsne = TSNE(n_jobs=8, n_components=2)
image = tsne.fit_transform(image)

print('Clustering....')
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, verbose=1).fit(image)
clust = kmeans.predict(image)
#np.save('clust.npy', clust)

print('Testing....')
test_data = pd.read_csv(test_path).values[:, 1:]

#clust = np.load('clust.npy')

y = []
for test in test_data:
    if clust[test[0]-1] == clust[test[1]-1]:
        y.append(1)
    else:
        y.append(0)

savePrediction(y, pred_path)

print('Predicttion time:', time()-start)
