import jieba
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, normalization, Flatten, BatchNormalization, Activation, Embedding, LSTM, GRU
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, LeakyReLU, PReLU, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from time import time
from keras.preprocessing import sequence
from gensim.models.word2vec import Word2Vec
import sys

train_x = sys.argv[1]
train_y = sys.argv[2]
test_x = sys.argv[3]
dict_path = sys.argv[4]

def Read_data(x_path, y_path):
	raw_data = pd.read_csv(x_path, encoding='utf-8')
	x = raw_data['comment'].values
	x = list([row for row in x])
	label = pd.read_csv(y_path)
	y = label['label'].values
	return x, y

def segment(x):
	seg_list = []
	for i in range(len(x)):
		seg_list.append(list(jieba.cut(x[i])))
	return seg_list

def Word2Index(x):
	dict_ = {}
	trans_x = []
	words_cnt = 0
	for i in range(len(x)):
		tmp = []
		for word in x[i]:
			if word in dict_:
				tmp.append(dict_[word])
			else:
				dict_[word] = words_cnt
				tmp.append(dict_[word])
				words_cnt += 1
		trans_x.append(tmp)
	return trans_x, words_cnt

if __name__ == '__main__':
	jieba.load_userdict(dict_path)
	start_time = time()
	x_train, y_train = Read_data(train_x, train_y)
	x_train = segment(x_train)
	x_train, words_cnt = Word2Index(x_train)
	print('Reading Time :', time() - start_time)

	model = Sequential()
	model.add(Embedding(words_cnt, output_dim=256, input_length=200))
	model.add(LSTM(96, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1))
	model.add(Activation("sigmoid"))
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	check_save  = ModelCheckpoint("./model_lstm/model-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)
	early_stop = EarlyStopping(monitor="val_loss", patience=3) 
	print(model.summary())

	start_time = time()
	x_train = sequence.pad_sequences(x_train, maxlen=200)
	train_history = model.fit(x_train, y_train, validation_split = 0.1, epochs = 2, batch_size = 256, verbose = 1, callbacks=[check_save,early_stop])
	'''plt.plot(train_history.history['acc'])
	plt.plot(train_history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('LSTM.png')
	print("Training Time :", time() - start_time)'''
	model.save('./model.h5')

