import numpy as np
import pandas as pd
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, normalization, Flatten, BatchNormalization, Activation, Embedding, LSTM
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, LeakyReLU, PReLU
import jieba
from time import time
from keras.preprocessing import sequence
import pickle

input_path = sys.argv[1]
dict_path = sys.argv[2]
out_path = sys.argv[3] 

def Read_data(x_path):
	raw_data = pd.read_csv(x_path, encoding='utf-8')
	x = raw_data['comment'].values
	x = list([row for row in x])
	return x

def segment(x):
	seg_list = []
	for i in range(len(x)):
		seg_list.append(list(jieba.cut(x[i])))
	return seg_list

def Word2Index(x_test):
	with open('dict.pickle', 'rb') as f:
		dict_ =pickle.load(f)
	#dict_ = {}
	trans_x = []
	'''words_cnt = 0
	for i in range(len(x)):
		for word in x[i]:
			if word not in dict_:
				dict_[word] = words_cnt
				words_cnt += 1
	f = open('dict.pickle', 'wb')
	pickle.dump(dict_, f)
	f.close()'''
	for i in range(len(x_test)):
		tmp = []
		for word in x_test[i]:
			if word in dict_:
				tmp.append(dict_[word])
			else:
				tmp.append(0)
		trans_x.append(tmp)
	return trans_x

def savePrediction(y, path, id_start=0):
    pd.DataFrame([[i+id_start, int(y[i])] for i in range(y.shape[0])],
                 columns=['id', 'label']).to_csv(path, index=False)

if __name__ == '__main__':
	jieba.load_userdict(dict_path)
	print("start reading...")
	start_time = time()
	x_test = Read_data(input_path)
	x_test = segment(x_test)
	x_test = Word2Index(x_test)
	print('Reading Time :', time() - start_time)

	print('start testing...')
	start_time = time()
	x_test = sequence.pad_sequences(x_test, maxlen=200)

	model_1 = load_model('model1.h5')
	model_2 = load_model('model2.h5')
	#model_3 = load_model('model3.h5')
	model_4 = load_model('model4.h5')
	model_5 = load_model('model5.h5')	

	y1 = model_1.predict(x_test)
	y2 = model_2.predict(x_test)
	#y3 = model_3.predict(x_test)
	y4 = model_4.predict(x_test)
	y5 = model_5.predict(x_test)

	y_test = []
	cnt_1 = 0
	for i in range(len(y1)):
		y = (y1[i]+y2[i]+y4[i]+y5[i])/4
		if (y >= 0.5):
			cnt_1 += 1
			y_test.append(1)
		else:
			y_test.append(0)
	y_test = np.array(y_test)

	savePrediction(y_test, out_path)
	print('Predict time :', time() - start_time)